"""
Tone Personas
=============
4 tone profiles that the email drafter uses to condition LLM output.
Each profile defines:
  - Description (for human review)
  - Salutation style
  - Closing style
  - LLM instruction modifier (steers the drafting model)
  - Length budget (words)
"""

TONE_PROFILES = {
    "friendly": {
        "description": "Warm, low-pressure, relationship-first",
        "salutation": "Hi {contact_first_name},",
        "closing": "Always great working with you.\n\nBest,\n{sender_name}",
        "llm_instruction": (
            "Use a warm, friendly tone. No pressure, no deadlines. Acknowledge it might already "
            "be in their processing queue. Ask politely for a confirmation or expected date. "
            "Keep it short — 3-4 sentences max. Indian English; respectful but informal."
        ),
        "max_words": 90,
    },
    "neutral": {
        "description": "Professional, factual, expects a response",
        "salutation": "Dear {contact_first_name},",
        "closing": "Looking forward to your prompt response.\n\nRegards,\n{sender_name}",
        "llm_instruction": (
            "Use a neutral professional tone. State the facts (invoice ID, amount, days "
            "overdue) clearly. Request a specific response (payment date or confirmation). "
            "No threats, no apologies. 4-5 sentences."
        ),
        "max_words": 130,
    },
    "firm": {
        "description": "Direct, urgent, references credit terms",
        "salutation": "Dear {contact_first_name},",
        "closing": "Please confirm payment by EOD this Friday or schedule a call.\n\nRegards,\n{sender_name}",
        "llm_instruction": (
            "Use a firm, direct tone. State invoice facts, mention this is now significantly "
            "overdue, reference credit terms (interest charges per contract). Include a deadline. "
            "Mention if a finance manager is being CC'd. 5-6 sentences. Indian English."
        ),
        "max_words": 170,
    },
    "final_notice": {
        "description": "Last formal communication before escalation",
        "salutation": "Dear {contact_first_name},",
        "closing": (
            "Failure to respond by [DATE] will result in escalation to our legal team and "
            "potential credit hold on future orders.\n\nRegards,\n{sender_name}"
        ),
        "llm_instruction": (
            "Use formal, final-notice tone. Reference history (number of prior reminders), "
            "specify exact deadline (7 days), mention specific consequences (credit hold, "
            "legal escalation, interest at 18% p.a.). Do NOT threaten or use accusatory "
            "language. State facts only. 6-7 sentences."
        ),
        "max_words": 220,
    },
}


# Map agent tool decisions to default tone
ACTION_TO_TONE = {
    "send_friendly_reminder": "friendly",
    "send_firm_escalation": "firm",
    "propose_payment_plan": "neutral",
    "flag_for_credit_hold": "firm",
    "escalate_to_legal": "final_notice",
    "wait": None,  # no email
}


def get_tone_for_action(action: str, segment: str) -> str:
    """
    Pick the tone based on the agent's action and customer segment.
    Strategic customers always get one tone level softer.
    """
    base_tone = ACTION_TO_TONE.get(action)
    if base_tone is None:
        return None

    # Strategic customers — soften one level
    if segment == "Strategic":
        soften = {"firm": "neutral", "final_notice": "firm"}
        base_tone = soften.get(base_tone, base_tone)
    # Bad Debt — harden one level
    if segment == "Bad Debt":
        harden = {"friendly": "neutral", "neutral": "firm", "firm": "final_notice"}
        base_tone = harden.get(base_tone, base_tone)

    return base_tone
