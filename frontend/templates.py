"""
HTML component templates for the Streamlit app.
"""


def hero_banner(title, subtitle):
    """Render the hero header banner."""
    return f"""
    <div class="hero">
        <span class="hero-badge">🛡️ Machine Learning Security</span>
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """


def risk_banner(kind, title, message):
    """
    Render a risk notification banner.
    kind: 'danger' | 'safe' | 'warning'
    """
    return f"""
    <div class="risk-banner {kind}">
        <h3>{title}</h3>
        <p>{message}</p>
    </div>
    """


def info_card(title, body_html):
    """Render a clean information card."""
    return f"""
    <div class="info-card">
        <h3>{title}</h3>
        {body_html}
    </div>
    """


def sidebar_model_card(label, model_name):
    """Render a sidebar status badge."""
    return f"""
    <div class="sidebar-model-card">
        <div>
            <div style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase;">{label}</div>
            <div class="sidebar-model-name">{model_name}</div>
        </div>
        <div class="sidebar-model-badge">READY</div>
    </div>
    """
