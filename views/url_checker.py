import re

import streamlit as st

from features.url_features import get_url_features, is_known_url_tld
from frontend.templates import risk_banner
from utils.prediction import get_phishing_prob, risk_label


def _result_box(pred, prob):
    confidence = prob if pred == 1 else 1 - prob
    risk = risk_label(prob)

    c1, c2, c3 = st.columns(3)
    c1.metric("Prediction", "Phishing URL" if pred == 1 else "Legitimate URL")
    c2.metric("Confidence", f"{confidence * 100:.2f}%")
    c3.metric("Risk Level", risk)

    if pred == 1:
        st.markdown(
            risk_banner(
                "danger",
                "Phishing URL Detected",
                "The prediction is based on structural characteristics commonly associated with phishing URLs. The indicators below contributed to the risk assessment.",
            ),
            unsafe_allow_html=True,
        )
        st.error("⚠️ Security Warning: Do not visit this URL, enter credentials, or download linked assets.")
    else:
        st.markdown(
            risk_banner(
                "safe",
                "Legitimate URL Detected",
                "No structural patterns commonly associated with phishing were identified in this URL. The risk assessment is based on learned feature patterns from the training dataset.",
            ),
            unsafe_allow_html=True,
        )
        st.info("💡 This URL passed structural screening. Always verify domain identity before entering personal information.")


def _build_url_explanation(feat_df, pred, url_input, trusted):
    """
    Generate human-readable explanations of why the URL was classified
    as phishing or legitimate, based on extracted feature values.
    """
    row = feat_df.iloc[0]
    suspicious = []
    safe_points = []

    # --- Suspicious indicators ---
    if int(row["IsDomainIP"]) == 1:
        suspicious.append((
            "🔴 IP Address Used as Domain",
            "This URL uses a raw IP address (e.g. http://192.168.1.1/login) instead of a proper "
            "domain name. Legitimate websites almost never use raw IP addresses. Attackers use this "
            "technique to avoid registering a traceable domain name."
        ))

    if int(row["IsHTTPS"]) == 0:
        suspicious.append((
            "🔴 No HTTPS Encryption",
            "This URL does not use HTTPS, meaning the connection is unencrypted. Any data you enter "
            "(passwords, card numbers) could be intercepted. All reputable websites use HTTPS today."
        ))

    if int(row["HasObfuscation"]) == 1:
        suspicious.append((
            "🔴 Character Obfuscation Detected",
            "The URL contains percent-encoded characters (e.g. %40, %2F) that are used to disguise "
            "its true destination. Phishing links often encode characters to bypass security filters "
            "and confuse users reading the link."
        ))

    url_len = float(row["URLLength"])
    if url_len > 75:
        suspicious.append((
            f"🔴 Abnormally Long URL ({int(url_len)} characters)",
            f"This URL is {int(url_len)} characters long. Phishing links are frequently padded with "
            "extra parameters or random strings to hide the real destination and make the link harder "
            "to read and analyse."
        ))

    subdomains = int(row["NoOfSubDomain"])
    if subdomains >= 3:
        suspicious.append((
            f"🔴 Excessive Subdomain Depth ({subdomains} subdomains)",
            f"This URL contains {subdomains} subdomains. Attackers often create deep subdomain chains "
            "like 'login.secure.verify.paypal.attacker.com' to make the link appear to belong to a "
            "trusted brand while the real domain is controlled by the attacker."
        ))

    if float(row["SpacialCharRatioInURL"]) > 0.20:
        ratio = float(row["SpacialCharRatioInURL"])
        suspicious.append((
            f"🔴 High Special Character Density ({ratio:.0%})",
            "The URL path or query string contains an unusually high proportion of special characters "
            "such as @, -, _, ~, and %. Legitimate URLs tend to be clean and readable. This pattern "
            "is commonly seen in obfuscated phishing links."
        ))

    if int(row["NoOfQMarkInURL"]) > 1:
        suspicious.append((
            "🔴 Multiple Query Delimiters (?)",
            "This URL contains more than one '?' character. Legitimate URLs use a single query string "
            "delimiter. Multiple '?' marks can indicate a malformed or deliberately confusing URL "
            "structure designed to bypass filters."
        ))

    if int(row["NoOfEqualsInURL"]) >= 3:
        eq_count = int(row["NoOfEqualsInURL"])
        suspicious.append((
            f"🔴 High Parameter Count ({eq_count} '=' signs)",
            f"The URL contains {eq_count} assignment operators in its query string. Phishing links "
            "frequently carry many hidden tracking or redirect parameters to route victims through "
            "multiple servers before reaching the fake login page."
        ))

    # --- Safe indicators ---
    if int(row["IsHTTPS"]) == 1:
        safe_points.append("✅ HTTPS encryption is active — the connection is secure.")

    if int(row["IsDomainIP"]) == 0:
        safe_points.append("✅ A proper domain name is used rather than a raw IP address.")

    if url_len <= 75:
        safe_points.append(f"✅ URL length is reasonable ({int(url_len)} characters) — not excessively padded.")

    if subdomains < 3:
        safe_points.append(f"✅ Subdomain depth is normal ({subdomains} subdomain(s) detected).")

    if int(row["HasObfuscation"]) == 0:
        safe_points.append("✅ No character obfuscation patterns detected in the URL.")

    if trusted:
        safe_points.append("✅ The domain uses a recognised trusted TLD (e.g. .ac.uk, .gov.uk, .edu, .de).")

    return suspicious, safe_points


def _awareness_section(pred, suspicious, safe_points):
    """Render the classification explanation and awareness guidance."""
    st.markdown("<br>", unsafe_allow_html=True)

    if pred == 1:
        st.subheader("🔍 Why Was This URL Flagged as Phishing?")
        st.markdown(
            "The prediction is based on structural characteristics commonly associated with phishing URLs. "
            "The **indicators below contributed to the risk assessment** — each one explains what was detected "
            "and why it is considered a risk signal."
        )
        st.caption(
            "ℹ️ Disclaimer: This tool performs offline structural text analysis only and does not "
            "visit the URL. Occasionally, legitimate websites with short domains or complex URL "
            "paths may be flagged incorrectly (false positive). If you recognise this as a trusted "
            "site, verify it directly in your browser address bar and check for the padlock icon."
        )

        if suspicious:
            for title, explanation in suspicious:
                with st.expander(title, expanded=True):
                    st.markdown(explanation)
        else:
            st.info(
                "The model's decision was driven by a combination of subtle feature patterns "
                "in the URL structure that collectively match phishing behaviour, even if no "
                "single indicator is highly alarming on its own."
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("🛡️ What Should You Do Now?")
        st.markdown("""
**Immediate Actions:**
- 🚫 **Do not visit this URL.** Close any tab where this link was opened.
- 🚫 **Do not enter any credentials** (username, password, bank details) on pages reached via this link.
- 🚫 **Do not download any files** linked from this URL — they may contain malware.
- 📧 **If received via email**, report the message to your IT or security team immediately.
- 🔗 **If you clicked the link**, run a malware scan on your device and change any passwords you may have entered.

**How to Protect Yourself:**
- Always check the full URL in your browser address bar before logging in anywhere.
- Look for the padlock icon and `https://` at the start of any website you trust.
- When in doubt, navigate directly to the official website by typing the address yourself.
- Enable two-factor authentication (2FA) on all important accounts so stolen passwords alone cannot grant access.
""")

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("📚 Phishing Awareness")
        st.info("""
**What is Phishing?**
Phishing is a type of cyber attack where criminals create fake websites or links that impersonate 
trusted organisations (banks, government services, online shops) to steal your login credentials, 
financial information, or personal data.

**Common Warning Signs:**
• Urgent language — "Your account will be suspended in 24 hours!"
• Requests for passwords, card numbers, or personal information via a link
• Slight misspellings in domain names (e.g. paypa1.com instead of paypal.com)
• Unexpected emails or messages asking you to verify your account
• Links that look different when you hover over them versus what is displayed
""")

    else:
        if safe_points:
            st.subheader("✅ Why This URL Appears Legitimate")
            st.markdown(
                "No significant structural patterns associated with phishing were detected in this URL. "
                "The following signals contributed to the low-risk assessment:"
            )
            for point in safe_points:
                st.success(point)

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("💡 Stay Safe — Even on Legitimate-Looking Sites")
        st.markdown("""
**General Cybersecurity Reminders:**
- ✅ Always verify the full domain name in your browser address bar before logging in.
- ✅ Check that the padlock icon is present — indicating an active HTTPS certificate.
- ✅ Be cautious of unexpected login prompts, even on sites that look familiar.
- ✅ Use a password manager — it will not auto-fill credentials on fake domains.
- ✅ Enable two-factor authentication (2FA) on your accounts for an extra layer of security.
- ✅ Keep your browser and operating system up to date to protect against known exploits.

> ⚠️ **Note:** This tool performs structural text analysis only. It does not visit the URL or 
> verify the live website content. Always apply your own judgement alongside the model's result.
""")


def render(model, feat_cols):
    st.subheader("🌐 URL Structural Threat Inspection")
    st.caption("Non-invasive lexical extraction — operates offline without initiating HTTP connections.")

    url_input = st.text_input(
        "Enter URL to Analyse",
        placeholder="e.g. https://secure-login.example.com/verify?token=abc123",
    )

    if not st.button("Analyse URL", type="primary", width="stretch"):
        return

    if not url_input.strip():
        st.warning("Please enter a URL before running the analysis.")
        return

    try:
        feats = get_url_features(url_input)
        feats = feats.reindex(columns=feat_cols)

        if feats.isnull().any().any():
            raise ValueError("Feature extraction failed for the input string.")

        pred = int(model.predict(feats)[0])
        prob = get_phishing_prob(model, feats)
        trusted = is_known_url_tld(url_input)

        _result_box(pred, prob)

        suspicious, safe_points = _build_url_explanation(feats, pred, url_input, trusted)
        _awareness_section(pred, suspicious, safe_points)

        with st.expander("🔍 Inspect Raw 18-Feature Vector"):
            st.dataframe(feats.T.rename(columns={0: "Feature Value"}), width="stretch")

    except Exception as err:
        st.error("Analysis encountered an error.")
        st.code(str(err))
