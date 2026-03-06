import os
from flask import Flask, render_template, request, redirect, url_for, session, flash
from supabase import create_client, Client
from anthropic import Anthropic
from dotenv import load_dotenv

# Load environment variables from .env (only used locally; Vercel uses its own env vars)
load_dotenv()

app = Flask(__name__)
# Flask uses this secret key to sign session cookies so they can't be tampered with
app.secret_key = os.environ.get("FLASK_SECRET_KEY")

# Initialise the Supabase client — this is our connection to the database and auth
supabase: Client = create_client(
    os.environ.get("SUPABASE_URL"),
    os.environ.get("SUPABASE_KEY"),
)

# Admin client uses the service_role key which bypasses Row Level Security.
# We only use this in therapist routes, AFTER verifying the therapist-client
# relationship in Python — never expose this key to the browser.
supabase_admin: Client = create_client(
    os.environ.get("SUPABASE_URL"),
    os.environ.get("SUPABASE_SERVICE_KEY"),
)

# Initialise the Anthropic client — this is our connection to Claude for mood analysis
anthropic = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_current_user():
    """Validate the stored access token with Supabase and return the user object."""
    token = session.get("access_token")
    if not token:
        return None
    try:
        response = supabase.auth.get_user(token)
        return response.user
    except Exception:
        return None


def get_role(user_id):
    """Look up this user's role ('client' or 'therapist') from user_profiles."""
    try:
        resp = (
            supabase_admin.table("user_profiles")
            .select("role")
            .eq("id", user_id)
            .maybe_single()
            .execute()
        )
        return resp.data["role"] if resp.data else "client"
    except Exception:
        return "client"


def require_therapist(user):
    """Return True if user exists and is a therapist; otherwise redirect."""
    if not user:
        return redirect(url_for("login"))
    if session.get("role") != "therapist":
        flash("Access restricted to therapists.", "error")
        return redirect(url_for("dashboard"))
    return None  # no redirect needed


# ---------------------------------------------------------------------------
# Routes — we'll fill in the logic for each one in later phases
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Landing page for visitors; redirect logged-in users straight to dashboard."""
    user = get_current_user()
    if user:
        return redirect(url_for("dashboard"))
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    """Dashboard — show mood trends for the logged-in user."""
    user = get_current_user()
    if not user:
        return redirect(url_for("login"))

    # Grab the 10 most recent entries to power the dashboard stats and timeline.
    # We only select the columns we actually need (not content) to keep it light.
    response = (
        supabase.table("journal_entries")
        .select("id, mood_label, mood_score, created_at, content")
        .eq("user_id", user.id)
        .order("created_at", desc=True)
        .limit(10)
        .execute()
    )
    entries = response.data

    # Calculate average mood score — only from entries that have one
    scored = [e for e in entries if e.get("mood_score") is not None]
    avg_score = round(sum(e["mood_score"] for e in scored) / len(scored), 1) if scored else None

    # Check for any pending therapist invites so we can show a banner
    pending_invites = (
        supabase_admin.table("therapist_clients")
        .select("id")
        .eq("client_email", user.email)
        .eq("status", "pending")
        .execute()
    ).data

    return render_template(
        "dashboard.html",
        user=user,
        entries=entries,
        avg_score=avg_score,
        pending_invites=len(pending_invites),
    )


@app.route("/login", methods=["GET", "POST"])
def login():
    """Show login form (GET) or process login (POST)."""
    if request.method == "POST":
        email    = request.form.get("email")
        password = request.form.get("password")
        try:
            # Ask Supabase to verify the email/password combination.
            # If correct, it returns a session object containing the access token.
            response = supabase.auth.sign_in_with_password(
                {"email": email, "password": password}
            )
            # Store the token in Flask's session cookie so we remember the user
            # across requests (HTTP is stateless by default).
            session["access_token"] = response.session.access_token
            role = get_role(response.user.id)
            session["role"] = role
            flash("Welcome back!", "success")
            # Send therapists to their portal, clients to their dashboard
            return redirect(url_for("therapist_dashboard") if role == "therapist" else url_for("dashboard"))
        except Exception as e:
            # Supabase raises an exception for bad credentials — catch it and
            # show the user a friendly error instead of a crash page.
            flash("Invalid email or password.", "error")

    return render_template("login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    """Show signup form (GET) or create a new account (POST)."""
    if request.method == "POST":
        email    = request.form.get("email")
        password = request.form.get("password")
        try:
            # Create a new user in Supabase Auth.
            # Supabase will send a confirmation email by default (you can turn
            # this off in Supabase Dashboard → Authentication → Settings).
            response = supabase.auth.sign_up(
                {"email": email, "password": password}
            )
            # After sign-up, Supabase may or may not return a session depending
            # on whether email confirmation is required.
            if response.session:
                session["access_token"] = response.session.access_token
                session["role"] = "client"
                # Create the user_profiles row so we know this is a client
                supabase_admin.table("user_profiles").insert(
                    {"id": response.user.id, "role": "client"}
                ).execute()
                flash("Account created! Welcome to Mood Journal.", "success")
                return redirect(url_for("dashboard"))
            else:
                # Email confirmation is enabled — tell the user to check inbox.
                flash("Check your email to confirm your account, then log in.", "info")
                return redirect(url_for("login"))
        except Exception as e:
            # Surface the real error so we can debug it
            flash(f"Could not create account: {str(e)}", "error")

    return render_template("signup.html")


@app.route("/logout")
def logout():
    """Clear the session and redirect to login."""
    session.clear()
    return redirect(url_for("login"))


@app.route("/journal")
def journal_list():
    """List all journal entries for the logged-in user, newest first."""
    user = get_current_user()
    if not user:
        return redirect(url_for("login"))

    # Query Supabase for all entries belonging to this user.
    # The RLS policy we set up means Supabase automatically filters by user —
    # but we still pass user_id so the query is explicit and readable.
    response = (
        supabase.table("journal_entries")
        .select("*")
        .eq("user_id", user.id)
        .order("created_at", desc=True)
        .execute()
    )
    entries = response.data  # list of dicts, one per row

    return render_template("journal_list.html", user=user, entries=entries)


@app.route("/journal/new", methods=["GET", "POST"])
def journal_new():
    """Show new-entry form (GET) or save entry + run mood analysis (POST)."""
    user = get_current_user()
    if not user:
        return redirect(url_for("login"))

    if request.method == "POST":
        content = request.form.get("content", "").strip()

        if not content:
            flash("Please write something before saving.", "error")
            return render_template("journal_new.html", user=user)

        # ── Mood analysis via Claude ────────────────────────────────────────
        # We send Claude the journal text and ask for structured output.
        # Asking for a specific format (label / score / summary) makes it easy
        # to parse the response reliably.
        try:
            ai_response = anthropic.messages.create(
                model="claude-3-5-haiku-20241022",  # fast + cheap model for this task
                max_tokens=300,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Analyse the mood of this journal entry. "
                            "Reply with ONLY these three lines, no extra text:\n"
                            "LABEL: <one or two word mood label, e.g. Anxious, Content, Excited>\n"
                            "SCORE: <integer 1-10 where 1=very negative, 10=very positive>\n"
                            "SUMMARY: <one sentence describing the mood>\n\n"
                            f"Journal entry:\n{content}"
                        ),
                    }
                ],
            )
            # Parse Claude's plain-text response into three variables
            raw = ai_response.content[0].text.strip()
            lines = {
                line.split(":")[0].strip(): line.split(":", 1)[1].strip()
                for line in raw.splitlines()
                if ":" in line
            }
            mood_label   = lines.get("LABEL", "Unknown")
            mood_score   = int(lines.get("SCORE", 5))
            mood_summary = lines.get("SUMMARY", "")
        except Exception:
            # If Claude fails for any reason, save the entry anyway with no mood data
            mood_label, mood_score, mood_summary = None, None, None

        # ── Save to Supabase ────────────────────────────────────────────────
        result = (
            supabase.table("journal_entries")
            .insert({
                "user_id":      user.id,
                "content":      content,
                "mood_label":   mood_label,
                "mood_score":   mood_score,
                "mood_summary": mood_summary,
            })
            .execute()
        )
        new_entry = result.data[0]  # Supabase returns the inserted row
        flash("Entry saved!", "success")
        return redirect(url_for("journal_detail", entry_id=new_entry["id"]))

    return render_template("journal_new.html", user=user)


@app.route("/journal/<entry_id>")
def journal_detail(entry_id):
    """Show a single journal entry with its mood analysis."""
    user = get_current_user()
    if not user:
        return redirect(url_for("login"))

    # Fetch the specific entry — .eq filters rows where id matches entry_id.
    # .maybe_single() returns one row or None (instead of a list).
    response = (
        supabase.table("journal_entries")
        .select("*")
        .eq("id", entry_id)
        .eq("user_id", user.id)   # extra safety: ensure it belongs to this user
        .maybe_single()
        .execute()
    )
    entry = response.data

    if not entry:
        flash("Entry not found.", "error")
        return redirect(url_for("journal_list"))

    return render_template("journal_detail.html", user=user, entry=entry)


@app.route("/journal/<entry_id>/delete", methods=["POST"])
def journal_delete(entry_id):
    """Delete a journal entry and redirect back to the list."""
    user = get_current_user()
    if not user:
        return redirect(url_for("login"))

    # We always filter by both id AND user_id so a user can never delete
    # someone else's entry by guessing an ID.
    supabase.table("journal_entries").delete().eq("id", entry_id).eq("user_id", user.id).execute()
    flash("Entry deleted.", "info")
    return redirect(url_for("journal_list"))


# ---------------------------------------------------------------------------
# Therapist signup
# ---------------------------------------------------------------------------

@app.route("/signup/therapist", methods=["GET", "POST"])
def signup_therapist():
    """Separate signup flow for licensed therapists."""
    if request.method == "POST":
        email    = request.form.get("email")
        password = request.form.get("password")
        name     = request.form.get("name", "").strip()
        try:
            response = supabase.auth.sign_up({"email": email, "password": password})
            if response.session:
                session["access_token"] = response.session.access_token
                session["role"] = "therapist"
                # Create user_profiles row with role='therapist'
                supabase_admin.table("user_profiles").insert({
                    "id":           response.user.id,
                    "role":         "therapist",
                    "display_name": name or email,
                }).execute()
                flash("Therapist account created!", "success")
                return redirect(url_for("therapist_dashboard"))
            else:
                flash("Check your email to confirm your account, then log in.", "info")
                return redirect(url_for("login"))
        except Exception as e:
            flash(f"Could not create account: {str(e)}", "error")

    return render_template("signup_therapist.html")


# ---------------------------------------------------------------------------
# Therapist portal
# ---------------------------------------------------------------------------

@app.route("/therapist")
def therapist_dashboard():
    """Therapist home: list linked clients and allow adding new ones."""
    user = get_current_user()
    blocked = require_therapist(user)
    if blocked:
        return blocked

    # Fetch all client links for this therapist
    resp = (
        supabase_admin.table("therapist_clients")
        .select("*")
        .eq("therapist_id", user.id)
        .order("created_at", desc=True)
        .execute()
    )
    clients = resp.data
    return render_template("therapist_dashboard.html", user=user, clients=clients)


@app.route("/therapist/add-client", methods=["POST"])
def therapist_add_client():
    """Add a client by email — creates a pending invite."""
    user = get_current_user()
    blocked = require_therapist(user)
    if blocked:
        return blocked

    client_email = request.form.get("client_email", "").strip().lower()
    if not client_email:
        flash("Please enter a client email.", "error")
        return redirect(url_for("therapist_dashboard"))

    try:
        supabase_admin.table("therapist_clients").insert({
            "therapist_id": user.id,
            "client_email": client_email,
            "status":       "pending",
        }).execute()
        flash(f"Invite sent to {client_email}. They will see it when they next log in.", "success")
    except Exception as e:
        # The unique constraint on (therapist_id, client_email) prevents duplicates
        flash("That client has already been invited.", "error")

    return redirect(url_for("therapist_dashboard"))


@app.route("/therapist/client/<client_id>/remove", methods=["POST"])
def therapist_remove_client(client_id):
    """Remove a client link entirely."""
    user = get_current_user()
    blocked = require_therapist(user)
    if blocked:
        return blocked

    supabase_admin.table("therapist_clients").delete().eq(
        "therapist_id", user.id
    ).eq("client_id", client_id).execute()
    flash("Client removed.", "info")
    return redirect(url_for("therapist_dashboard"))


@app.route("/therapist/client/<client_id>/report")
def therapist_report(client_id):
    """
    Mood report for one client over a chosen period.
    Therapist sees scores, labels and AI summaries — never raw journal text.
    """
    user = get_current_user()
    blocked = require_therapist(user)
    if blocked:
        return blocked

    # Verify this therapist is actually linked to this client
    link = (
        supabase_admin.table("therapist_clients")
        .select("*")
        .eq("therapist_id", user.id)
        .eq("client_id", client_id)
        .eq("status", "accepted")
        .maybe_single()
        .execute()
    )
    if not link.data:
        flash("Client not found or not yet accepted your invite.", "error")
        return redirect(url_for("therapist_dashboard"))

    # Period selector — defaults to weekly
    period = request.args.get("period", "weekly")
    days   = {"weekly": 7, "biweekly": 14, "monthly": 30}.get(period, 7)
    from datetime import datetime, timedelta, timezone
    period_start = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    # Fetch mood data only — content column deliberately excluded
    entries_resp = (
        supabase_admin.table("journal_entries")
        .select("id, mood_label, mood_score, mood_summary, created_at")
        .eq("user_id", client_id)
        .gte("created_at", period_start)
        .order("created_at")
        .execute()
    )
    entries = entries_resp.data

    # ── Compute stats ────────────────────────────────────────────────────
    scored = [e for e in entries if e.get("mood_score") is not None]
    avg_score = round(sum(e["mood_score"] for e in scored) / len(scored), 1) if scored else None

    # Label frequency: {"Anxious": 3, "Content": 2, ...}
    from collections import Counter
    label_counts = dict(Counter(
        e["mood_label"] for e in entries if e.get("mood_label")
    ))

    # Trend: compare first-half avg vs second-half avg
    trend = None
    if len(scored) >= 4:
        mid       = len(scored) // 2
        first_avg = sum(e["mood_score"] for e in scored[:mid]) / mid
        second_avg= sum(e["mood_score"] for e in scored[mid:]) / (len(scored) - mid)
        diff      = second_avg - first_avg
        trend = "improving" if diff > 0.5 else "declining" if diff < -0.5 else "stable"

    # ── Claude narrative summary ─────────────────────────────────────────
    # We summarise using only mood metadata, not journal text, to preserve privacy.
    ai_summary = None
    if entries:
        try:
            mood_data = "\n".join(
                f"- {e['created_at'][:10]}: {e.get('mood_label','?')} "
                f"(score {e.get('mood_score','?')}): {e.get('mood_summary','')}"
                for e in entries
            )
            ai_resp = anthropic.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=250,
                messages=[{
                    "role": "user",
                    "content": (
                        f"You are assisting a licensed therapist. Based ONLY on the following "
                        f"mood tracking data from the past {days} days, write a 2-3 sentence "
                        f"clinical summary suitable for a therapy session. "
                        f"Do not make assumptions beyond what the data shows.\n\n"
                        f"{mood_data}"
                    )
                }]
            )
            ai_summary = ai_resp.content[0].text.strip()
        except Exception:
            ai_summary = None

    return render_template(
        "therapist_report.html",
        user=user,
        client_id=client_id,
        client_email=link.data["client_email"],
        entries=entries,
        avg_score=avg_score,
        label_counts=label_counts,
        trend=trend,
        ai_summary=ai_summary,
        period=period,
        days=days,
    )


# ---------------------------------------------------------------------------
# Client invite management
# ---------------------------------------------------------------------------

@app.route("/client/invites")
def client_invites():
    """Show pending therapist invites for the logged-in client."""
    user = get_current_user()
    if not user:
        return redirect(url_for("login"))

    # Find any pending invites addressed to this user's email
    resp = (
        supabase_admin.table("therapist_clients")
        .select("*")
        .eq("client_email", user.email)
        .eq("status", "pending")
        .execute()
    )
    invites = resp.data
    return render_template("client_invites.html", user=user, invites=invites)


@app.route("/client/invites/<invite_id>/accept", methods=["POST"])
def client_invite_accept(invite_id):
    """Accept a therapist invite — sets status to accepted and records client_id."""
    user = get_current_user()
    if not user:
        return redirect(url_for("login"))

    supabase_admin.table("therapist_clients").update({
        "status":    "accepted",
        "client_id": user.id,
    }).eq("id", invite_id).eq("client_email", user.email).execute()
    flash("Therapist connection accepted.", "success")
    return redirect(url_for("dashboard"))


@app.route("/client/invites/<invite_id>/decline", methods=["POST"])
def client_invite_decline(invite_id):
    """Decline a therapist invite — deletes the row."""
    user = get_current_user()
    if not user:
        return redirect(url_for("login"))

    supabase_admin.table("therapist_clients").delete().eq(
        "id", invite_id
    ).eq("client_email", user.email).execute()
    flash("Invite declined.", "info")
    return redirect(url_for("dashboard"))


# ---------------------------------------------------------------------------
# Run locally with `python app.py`
# On Vercel, it imports the `app` object directly (no __main__ block needed)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
