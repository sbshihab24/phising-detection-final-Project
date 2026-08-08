import re

import streamlit as st

from features.email_features import clean_text
from frontend.templates import risk_banner
from utils.prediction import get_phishing_prob, risk_label


# Phishing trigger phrases commonly found in social engineering emails
PHISHING_PHRASES = [
    (r"\baccount.{0,20}(suspend|terminat|block|lock|clos)", "Account suspension threat"),
    (r"\b(verify|confirm|validate).{0,20}(account|identity|email|details)", "Identity/account verification request"),
    (r"\bclick.{0,20}(here|below|link|button).{0,30}(now|immediately|urgent)", "Urgent click-through instruction"),
    (r"\b(urgent|immediate|action required|respond within)", "Urgency and pressure language"),
    (r"\b(password|credential|login).{0,20}(reset|update|enter|provide)", "Credential harvesting language"),
    (r"\b(won|winner|prize|reward|lottery|selected|congratulation)", "Prize or reward lure"),
    (r"\b(bank|paypal|amazon|apple|microsoft|google|netflix).{0,30}(account|security|login)", "Impersonation of a trusted brand"),
    (r"\b(social security|ssn|national insurance|date of birth|mother.s maiden)", "Request for sensitive personal information"),
    (r"\b(wire transfer|western union|gift card|itunes|bitcoin|cryptocurrency).{0,30}(pay|send|purchase)", "Unusual payment method request"),
    (r"\bdo not (share|tell|inform|show) (this|anyone)", "Secrecy instruction — common in scams"),
]

SAFE_SIGNALS = [
    (r"\b(unsubscribe|preferences|manage subscriptions)", "Contains standard unsubscribe footer (newsletter pattern)"),
    (r"\b(regards|sincerely|best wishes|kind regards|thank you for your)", "Professional closing language detected"),
    (r"\b(meeting|agenda|attached|report|project|update|schedule)", "Typical business communication vocabulary"),
    (r"\b(invoice|receipt|order confirmation|tracking number)", "Legitimate transactional email language"),
]


def _result_box(pred, prob):
    confidence = prob if pred == 1 else 1 - prob
    risk = risk_label(prob)

    c1, c2, c3 = st.columns(3)
    c1.metric("Prediction", "Phishing Email" if pred == 1 else "Safe Email")
    c2.metric("Confidence", f"{confidence * 100:.2f}%")
    c3.metric("Risk Level", risk)

    if pred == 1:
        st.markdown(
            risk_banner(
                "danger",
                "Phishing Email Detected",
                "The prediction is based on language patterns statistically associated with phishing and social engineering emails.",
            ),
            unsafe_allow_html=True,
        )
        st.error("⚠️ Security Warning: Do not click embedded links, open attachments, or reply with confidential credentials.")
    else:
        st.markdown(
            risk_banner(
                "safe",
                "Safe Email Detected",
                "No significant phishing language patterns were detected. The risk assessment is based on learned vocabulary distributions from the training dataset.",
            ),
            unsafe_allow_html=True,
        )
        st.info("💡 This email passed NLP screening. Always verify the sender domain before clicking any links.")


def _scan_email_text(raw_text):
    """Scan raw email text for known phishing phrases and safe signals."""
    text_lower = raw_text.lower()
    found_phishing = []
    found_safe = []

    for pattern, label in PHISHING_PHRASES:
        if re.search(pattern, text_lower):
            found_phishing.append(label)

    for pattern, label in SAFE_SIGNALS:
        if re.search(pattern, text_lower):
            found_safe.append(label)

    return found_phishing, found_safe


def _awareness_section(pred, raw_text):
    """Render classification explanation and user awareness section."""
    found_phishing, found_safe = _scan_email_text(raw_text)

    st.markdown("<br>", unsafe_allow_html=True)

    if pred == 1:
        st.subheader("🔍 Why Was This Email Flagged as Phishing?")
        st.markdown(
            "The NLP classifier analysed the vocabulary and language structure of this email against "
            "patterns learned from thousands of known phishing and legitimate emails. "
            "The following **risk signals were detected** in the text:"
        )

        if found_phishing:
            st.markdown("---")
            for i, label in enumerate(found_phishing, 1):
                st.markdown(f"**{i}. {label}**")
                _explain_phishing_phrase(label)
                st.markdown("---")
        else:
            st.markdown("---")
            st.markdown("**1. Overall Language Pattern — High Phishing Resemblance**")
            st.markdown(
                "No single trigger phrase was matched, but the **combined vocabulary, tone, and sentence "
                "structure** of this email closely resembles known phishing emails in the training dataset. "
                "NLP classifiers detect phishing through the statistical distribution of all words — not "
                "just individual keywords. Subtle signals such as urgency tone, unusual phrasing, and "
                "atypical word combinations all contribute to the risk score."
            )
            st.markdown("---")
            st.markdown("**2. Recommendation — Apply Caution**")
            st.markdown(
                "Even when no explicit phishing keyword is found, the model's confidence score indicates "
                "this email's language is statistically unusual compared to normal communication. "
                "Treat it with the same caution you would apply to any suspicious message."
            )
            st.markdown("---")

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("🛡️ What Should You Do Now?")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🚫 Immediate Actions:**")
            st.markdown("""
1. **Do not click any links** — even if they appear to go to a trusted website.
2. **Do not open attachments** — files such as .zip, .exe, .pdf, or .docm may contain malware.
3. **Do not reply** with any personal information, passwords, or financial details.
4. **Report the email** to your IT department or email provider.
5. **Delete the email** from your inbox and trash folder after reporting.
""")
        with col2:
            st.markdown("**🔍 How to Verify the Sender:**")
            st.markdown("""
1. Check the **actual sender email address** — not just the display name shown in your inbox.
2. Phishers often display "Amazon Support" but send from a random Gmail or unknown domain address.
3. If unsure, contact the organisation **directly** using contact details from their official website — never from the email itself.
4. **If you already clicked a link or entered details**, change your passwords immediately and enable two-factor authentication (2FA).
""")

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("📚 Email Phishing Awareness")

        st.markdown("**What is Email Phishing?**")
        st.markdown(
            "Email phishing is the most common form of cyber attack. Criminals impersonate banks, "
            "online services, employers, or government agencies to trick you into revealing passwords, "
            "financial details, or installing malware on your device."
        )

        st.markdown("**Common Warning Signs to Look For:**")
        signs = [
            "Creates a sense of urgency — *'Act now or your account will be closed!'*",
            "Asks you to click a link and log in, even if you did not request a reset.",
            "The sender email address looks slightly wrong (e.g. support@paypa1.com instead of paypal.com).",
            "Contains unusual spelling or grammar errors for a professional organisation.",
            "Offers something too good to be true — prizes, refunds, or unexpected windfalls.",
            "Asks you to keep the email confidential or act without telling anyone.",
        ]
        for i, sign in enumerate(signs, 1):
            st.markdown(f"{i}. {sign}")

        st.info(
            "🔑 **The Golden Rule:** No legitimate bank, government service, or reputable company "
            "will ever ask for your password via email."
        )



    else:
        st.subheader("✅ Why This Email Appears Safe")
        st.markdown(
            "The NLP model found the language in this email consistent with normal, legitimate communication. "
            "Here are the positive signals detected:"
        )

        if found_safe:
            for label in found_safe:
                st.success(f"✅ {label}")
        else:
            st.success(
                "✅ The overall word patterns and vocabulary in this email are statistically consistent "
                "with safe, non-phishing communication. No high-risk language triggers were found."
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("💡 Stay Vigilant — Even on Safe-Looking Emails")
        st.markdown("""
**General Email Safety Reminders:**
- ✅ Always check the sender's full email address, not just the display name.
- ✅ Hover over links before clicking — check that the URL matches the claimed destination.
- ✅ Be cautious of unexpected emails asking you to log in or confirm personal details.
- ✅ Never send passwords, bank details, or sensitive information by email.
- ✅ Enable spam filters and keep your email client and antivirus software up to date.
- ✅ Use two-factor authentication (2FA) so stolen passwords alone cannot compromise your accounts.

> ⚠️ **Note:** This tool analyses the text content of the email only. It does not scan 
> attachments, headers, or embedded links. Always exercise caution with unexpected emails.
""")


def _explain_phishing_phrase(label):
    """Provide a detailed explanation for each detected phishing signal."""
    explanations = {
        "Account suspension threat": (
            "This email uses the threat of account closure or suspension to create panic and pressure "
            "you into acting quickly without thinking. This is one of the most common phishing tactics — "
            "legitimate companies give clear, calm notice through official channels and never demand "
            "immediate action via an email link."
        ),
        "Identity/account verification request": (
            "The email asks you to verify your identity or account details through a link. Phishing "
            "emails routinely direct victims to fake login pages that look identical to real websites. "
            "Once you enter your credentials, they are captured by the attacker. Legitimate services "
            "do not ask you to verify your account via an unsolicited email link."
        ),
        "Urgent click-through instruction": (
            "The email instructs you to click a link immediately or urgently. This urgency is "
            "deliberate — it is designed to override your caution and stop you from thinking critically "
            "about whether the email is genuine. Slow down and verify before clicking anything."
        ),
        "Urgency and pressure language": (
            "Phrases like 'urgent', 'immediate action required', or 'respond within 24 hours' are "
            "classic social engineering tactics. Attackers use time pressure to stop you from checking "
            "whether the email is real. If an email demands instant action, treat it with extra suspicion."
        ),
        "Credential harvesting language": (
            "This email references passwords, login credentials, or asks you to 'update' or 'reset' "
            "your details. This is a credential harvesting attempt — the attacker wants your login "
            "information. No legitimate organisation will ask you to enter your password via an email link."
        ),
        "Prize or reward lure": (
            "This email claims you have won something — a prize, lottery, reward, or have been "
            "specially selected. This is a well-known social engineering lure. The goal is to get you "
            "excited enough to click a link or provide personal information. If you did not enter a "
            "competition, you cannot have won it."
        ),
        "Impersonation of a trusted brand": (
            "The email references a well-known organisation such as a bank, Amazon, PayPal, Apple, "
            "Microsoft, or a similar trusted brand. Attackers impersonate these companies because "
            "people trust them and are more likely to follow their instructions. Always verify by "
            "going directly to the official website rather than clicking the email link."
        ),
        "Request for sensitive personal information": (
            "The email requests highly sensitive personal information such as your Social Security "
            "Number, National Insurance number, date of birth, or answers to security questions. "
            "This information is used for identity theft. No legitimate service needs this information "
            "sent via email."
        ),
        "Unusual payment method request": (
            "The email asks for payment via an unusual method such as gift cards, wire transfer, "
            "cryptocurrency, or cash. These payment methods are irreversible and untraceable, which "
            "is exactly why scammers prefer them. Legitimate businesses do not ask for payment via "
            "iTunes gift cards or Bitcoin."
        ),
        "Secrecy instruction — common in scams": (
            "The email instructs you to keep it confidential or not tell anyone about it. This is a "
            "manipulation tactic designed to prevent you from checking with a trusted person who might "
            "recognise the scam. Legitimate communications never ask you to keep them secret."
        ),
    }
    text = explanations.get(label, "This pattern is commonly associated with phishing or social engineering emails.")
    st.markdown(text)


def render(model, vectorizer):
    st.subheader("📧 Email Content NLP Analysis")
    st.caption("Submissions are tokenised into 10,000 TF-IDF features and classified locally in session memory.")

    email_input = st.text_area(
        "Paste Email Body Text",
        height=260,
        placeholder=(
            "Paste the full email body text here...\n"
            "e.g. Dear customer, your account requires immediate verification. "
            "Please click the secure link below within 24 hours to prevent suspension."
        ),
    )

    if not st.button("Analyse Email", type="primary", width="stretch"):
        return

    if not email_input.strip():
        st.warning("Please paste email body text before running the analysis.")
        return

    try:
        cleaned = clean_text(email_input)

        if not cleaned:
            raise ValueError("No readable words remained after text pre-cleaning.")

        X = vectorizer.transform([cleaned])
        pred = int(model.predict(X)[0])
        prob = get_phishing_prob(model, X)

        _result_box(pred, prob)
        _awareness_section(pred, email_input)

        with st.expander("🔍 View Pre-Processed NLP Text"):
            st.code(cleaned)

    except Exception as err:
        st.error("Analysis encountered an error.")
        st.code(str(err))
