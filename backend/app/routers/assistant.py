import re
from datetime import datetime
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends
from sqlalchemy import select, func, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.middleware.auth import get_current_user, get_user_mine_ids, check_mine_access
from app.models.user import User, UserRole
from app.models.mine import Mine
from app.models.compliance import Compliance, ComplianceStatus, RiskLevel
from app.models.observation import Observation, ObservationSeverity, ObservationStatus
from app.models.inspection import Inspection
from app.models.contractor import Contractor
from app.services.llm_service import LLMService
from app.schemas.assistant import AssistantChatRequest, AssistantChatResponse, QueryQuickAction

router = APIRouter()


def is_hindi(text: str) -> bool:
    """Detect if text contains Devanagari Unicode characters."""
    return any('ऀ' <= char <= 'ॿ' for char in text)


def format_hindi(text: str) -> str:
    """Ensure text is properly displayed in Hindi."""
    return text


@router.post("/chat", response_model=AssistantChatResponse)
async def chat_with_assistant(
    request: AssistantChatRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> AssistantChatResponse:
    """
    Multilingual Conversational AI Assistant for CoalGuard AI (Section 15 PRD).
    Uses an optional LLM (OpenAI/Anthropic/Ollama) with RAG retrieval for natural
    language responses in English and Hindi. Falls back to rule-based responses
    if no LLM is configured.
    """
    query = request.message.strip().lower()
    user_lang = request.language
    if user_lang == "auto" or not user_lang:
        user_lang = "hi" if is_hindi(request.message) else "en"

    user_mine_ids = await get_user_mine_ids(db, current_user.id)
    is_admin_or_corp = current_user.role in [UserRole.admin, UserRole.corporate, UserRole.regulatory]

    # Target specific mine or user mines
    filter_mine_id = request.mine_id
    if filter_mine_id and not is_admin_or_corp and filter_mine_id not in user_mine_ids:
        filter_mine_id = user_mine_ids[0] if user_mine_ids else None

    # Initialize LLM Service
    llm_service = LLMService()

    # Retrieve basic context for RAG
    mines_stmt = select(Mine)
    if not is_admin_or_corp and user_mine_ids:
        mines_stmt = mines_stmt.where(Mine.id.in_(user_mine_ids))
    mines_res = await db.execute(mines_stmt)
    accessible_mines = mines_res.scalars().all()
    mine_map = {m.id: m.name for m in accessible_mines}

    quick_actions: List[QueryQuickAction] = []
    data_context: Dict[str, Any] = {}
    llm_used = False

    # Build RAG Context Data from Database
    context_parts: List[str] = []

    # Mine Context
    if accessible_mines:
        mine_lines = []
        for m in accessible_mines:
            mine_lines.append(f"  - {m.name} (State: {m.state}, Subsidiary: {m.subsidiary}, Status: {m.status})")
        context_parts.append("Mines Overview:\n" + "\n".join(mine_lines))

    # Compliance Context (only overdue for performance)
    comp_stmt = select(Compliance).where(Compliance.status == ComplianceStatus.overdue)
    if filter_mine_id:
        comp_stmt = comp_stmt.where(Compliance.mine_id == filter_mine_id)
    elif not is_admin_or_corp and user_mine_ids:
        comp_stmt = comp_stmt.where(Compliance.mine_id.in_(user_mine_ids))
    comp_res = await db.execute(comp_stmt)
    overdue_items = comp_res.scalars().all()

    if overdue_items:
        comp_lines = []
        for idx, item in enumerate(overdue_items[:10], 1):
            comp_lines.append(
                f"  {idx}. [{mine_map.get(item.mine_id, 'Mine')}] {item.requirement} (Category: {item.category}, Due: {item.due_date}, Risk: {item.risk_level})"
            )
        context_parts.append("Overdue Compliance Items:\n" + "\n".join(comp_lines))
        data_context["overdue_count"] = len(overdue_items)
        data_context["overdue_items"] = [
            {
                "id": item.id,
                "mine": mine_map.get(item.mine_id, "Mine"),
                "requirement": item.requirement,
                "category": item.category,
                "due_date": item.due_date.isoformat() if item.due_date else None,
                "risk_level": item.risk_level.value if hasattr(item.risk_level, 'value') else item.risk_level,
            }
            for item in overdue_items[:5]
        ]
        quick_actions.append(QueryQuickAction(label="Open Compliance Register", action_type="navigate", payload="/mine/compliance"))

    # Observations Context (High Risk only)
    obs_stmt = select(Observation).where(
        Observation.severity.in_([ObservationSeverity.critical, ObservationSeverity.high]),
        Observation.status.in_([ObservationStatus.open, ObservationStatus.in_progress]),
    )
    obs_res = await db.execute(obs_stmt)
    high_risk_obs = obs_res.scalars().all()

    if high_risk_obs:
        obs_lines = []
        for idx, o in enumerate(high_risk_obs[:10], 1):
            obs_lines.append(f"  {idx}. [{o.severity}] {o.description} (Status: {o.status})")
        context_parts.append("Active High/Critical Safety Observations:\n" + "\n".join(obs_lines))
        data_context["high_risk_count"] = len(high_risk_obs)
        data_context["observations"] = [
            {
                "id": o.id,
                "description": o.description,
                "severity": o.severity.value if hasattr(o.severity, 'value') else o.severity,
                "status": o.status.value if hasattr(o.status, 'value') else o.status,
            }
            for o in high_risk_obs[:5]
        ]
        quick_actions.append(QueryQuickAction(label="View Hazard Log", action_type="navigate", payload="/mine/observations"))
        quick_actions.append(QueryQuickAction(label="Open Risk Map", action_type="navigate", payload="/map"))

    # Inspections Context
    insp_stmt = select(Inspection).order_by(Inspection.date.desc()).limit(5)
    if filter_mine_id:
        insp_stmt = insp_stmt.where(Inspection.mine_id == filter_mine_id)
    insp_res = await db.execute(insp_stmt)
    inspections = insp_res.scalars().all()

    if inspections:
        insp_lines = []
        for idx, insp in enumerate(inspections, 1):
            insp_lines.append(
                f"  {idx}. [{mine_map.get(insp.mine_id, 'Mine')}] {insp.inspection_type.capitalize()} Audit — Date: {insp.date.strftime('%d %b %Y')} (Status: {insp.status})"
            )
        context_parts.append("Recent Statutory Inspections:\n" + "\n".join(insp_lines))

    # Contractors Context
    if is_admin_or_corp or user_mine_ids:
        contractor_stmt = select(Contractor)
        if filter_mine_id:
            contractor_stmt = contractor_stmt.where(Contractor.mine_id == filter_mine_id)
        elif not is_admin_or_corp and user_mine_ids:
            contractor_stmt = contractor_stmt.where(Contractor.mine_id.in_(user_mine_ids))
        contractor_res = await db.execute(contractor_stmt)
        contractors = contractor_res.scalars().all()
        if contractors:
            contractor_lines = []
            for c in contractors[:5]:
                contractor_lines.append(
                    f"  - {c.name} (Workers: {c.worker_count}, Compliance: {c.compliance_status}, Contract End: {c.contract_end})"
                )
            context_parts.append("Contractor Overview:\n" + "\n".join(contractor_lines))

    rag_context = "\n\n".join(context_parts) if context_parts else None

    # ---- Intent Detection (Rule-Based Fallback / Keywords) ----
    is_overdue_query = any(k in query for k in ["overdue", "due", "विलंबित", "बकाया", "deadline", "समय सीमा", "अनुपालन"])
    is_hazard_query = any(k in query for k in ["high risk", "critical", "hazard", "observation", "जोखिम", "आपत्ति", "खतरा", "गंभीर"])
    is_inspection_query = any(k in query for k in ["inspection", "audit", "निरीक्षण", "ऑडिट", "जांच"])
    is_summary_query = any(k in query for k in ["summary", "status", "mine", "खदान", "स्थिति", "डैशबोर्ड", "dashboard", "report"])

    # ---- LLM-Enhanced Response (If Available) ----
    if llm_service.is_available():
        llm_used = True
        try:
            llm_reply = await llm_service.chat_completion(
                user_message=request.message,
                context_data=rag_context,
                conversation_history=[
                    {"role": m.role, "content": m.content} for m in (request.conversation_history or [])
                ],
            )
            if llm_reply:
                return AssistantChatResponse(
                    reply=llm_reply,
                    language=user_lang,
                    data_context=data_context,
                    quick_actions=quick_actions,
                    used_llm=True,
                )
        except Exception as e:
            import logging as _logging
            _logging.getLogger(__name__).warning(f"LLM fallback triggered: {e}")

    # ---- Rule-Based Fallback Responses ----
    if is_overdue_query:
        if not overdue_items:
            reply = "✅ Great news! There are currently **no overdue compliance items** across your assigned coal leases."
        else:
            reply = f"⚠️ Found **{len(overdue_items)} overdue compliance obligations** requiring attention:\n\n"
            for idx, itm in enumerate(overdue_items[:4], 1):
                m_name = mine_map.get(itm.mine_id, "Mine")
                reply += f"{idx}. **[{m_name}]** {itm.requirement} — *Category: {itm.category}*\n"
            if len(overdue_items) > 4:
                reply += f"\n...and {len(overdue_items) - 4} more items. View all in the Compliance Register."
        if user_lang == "hi":
            reply = "✅ बहुत बढ़िया! अभी कोई वैधानिक अनुपालन (Compliance) लंबित नहीं है।" if not overdue_items else (
                f"⚠️ **{len(overdue_items)} वैधानिक अनुपालन** समय सीमा पार कर चुके हैं:\n\n"
                + "\n".join(f"{idx}. **[{mine_map.get(itm.mine_id, 'खदान')}]** {itm.requirement} (श्रेणी: *{itm.category}*)" for idx, itm in enumerate(overdue_items[:4], 1))
                + (f"\n...और {len(overdue_items) - 4} अन्य।" if len(overdue_items) > 4 else "")
            )

    elif is_hazard_query:
        if not high_risk_obs:
            reply = "🛡️ Excellent! There are currently **no open high-risk or critical safety hazards** logged."
        else:
            reply = f"🚨 **{len(high_risk_obs)} active high-risk / critical safety observations** require intervention:\n\n"
            for idx, o in enumerate(high_risk_obs[:4], 1):
                reply += f"{idx}. **[{o.severity.upper()}]** {o.description} (*Status: {o.status}*)\n"
            reply += "\nAssign corrective actions from the Hazard Tracking log."
        if user_lang == "hi":
            reply = "🛡️ बहुत अच्छा! कोई खुला उच्च-जोखिम खतरा नहीं है।" if not high_risk_obs else (
                f"🚨 **{len(high_risk_obs)} उच्च-जोखिम वाली आपत्तियां** सक्रिय हैं:\n\n"
                + "\n".join(f"{idx}. **[{o.severity.upper()}]** {o.description} (स्थिति: *{o.status}*)" for idx, o in enumerate(high_risk_obs[:4], 1))
                + "\nसुधारात्मक कार्रवाई असाइन करें।"
            )

    elif is_inspection_query:
        reply = f"📋 Here are the **{len(inspections)} most recent statutory inspections**:\n\n"
        for idx, insp in enumerate(inspections, 1):
            m_name = mine_map.get(insp.mine_id, "Mine")
            reply += f"{idx}. **[{m_name}]** {insp.inspection_type.capitalize()} Audit — Date: {insp.date.strftime('%d %b %Y')} (*Status: {insp.status}*)\n"
        if user_lang == "hi":
            reply = f"📋 **हाल के {len(inspections)} निरीक्षण** का विवरण:\n\n" + "\n".join(
                f"{idx}. **[{mine_map.get(insp.mine_id, 'खदान')}]** {insp.inspection_type.capitalize()} ऑडिट — {insp.date.strftime('%d %b %Y')} ({insp.status})"
                for idx, insp in enumerate(inspections, 1)
            )

    elif is_summary_query:
        total_mines = len(accessible_mines)
        comp_count_res = await db.execute(select(func.count(Compliance.id)))
        total_comp = comp_count_res.scalar() or 0
        obs_count_res = await db.execute(select(func.count(Observation.id)))
        total_obs = obs_count_res.scalar() or 0
        reply = f"🏢 **CoalGuard AI Operational Summary:**\n\n• 📍 **Active Mines:** {total_mines}\n• 📑 **Statutory Records:** {total_comp}\n• ⚠️ **Safety Observations:** {total_obs}\n\nAsk me about overdue compliance, critical hazards, or recent inspections."
        if user_lang == "hi":
            reply = f"🏢 **कोल-गार्ड AI सारांश:**\n\n• 📍 **सक्रिय खदानें:** {total_mines}\n• 📑 **रिकॉर्ड:** {total_comp}\n• ⚠️ **आपत्तियां:** {total_obs}\n\nमुझसे लंबित अनुपालन या उच्च-जोखिम वाले खतरों के बारे में पूछें।"

    else:
        quick_actions = [
            QueryQuickAction(label="Show Overdue Compliance", action_type="query", payload="Show overdue compliance items"),
            QueryQuickAction(label="High Risk Observations", action_type="query", payload="Which mines have high-risk observations?"),
            QueryQuickAction(label="Recent Inspections", action_type="query", payload="Show recent inspections"),
            QueryQuickAction(label="हिन्दी में सारांश", action_type="query", payload="खदान की वर्तमान स्थिति और जोखिम का सारांश दें"),
        ]
        if user_lang == "hi":
            reply = "नमस्ते! मैं **CoalGuard AI** का बहुभाषी सहायक हूँ। मैं अनुपालन, सुरक्षा निरीक्षण, और उच्च-जोखिम वाले खतरों के बारे में अंग्रेजी या हिंदी में जवाब दे सकता हूँ।"
        else:
            reply = "Hello! I am the **CoalGuard AI** Smart Assistant. I can analyze statutory compliance, safety hazards, audit checklists, and DGMS field regulations in English or Hindi. Try clicking a suggested action below!"

    return AssistantChatResponse(
        reply=reply,
        language=user_lang,
        data_context=data_context,
        quick_actions=quick_actions,
        used_llm=llm_used,
    )
