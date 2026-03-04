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

# Initialise the Anthropic client — this is our connection to Claude for mood analysis
anthropic = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


# ---------------------------------------------------------------------------
# Helper: check if the user is logged in
# We store their Supabase access token in Flask's session after login.
# ---------------------------------------------------------------------------
def get_current_user():
    token = session.get("access_token")
    if not token:
        return None
    try:
        # Ask Supabase to validate the token and return the user object
        response = supabase.auth.get_user(token)
        return response.user
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Routes — we'll fill in the logic for each one in later phases
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Dashboard — show mood trends. Redirects to login if not authenticated."""
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

    return render_template("index.html", user=user, entries=entries, avg_score=avg_score)


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
            flash("Welcome back!", "success")
            return redirect(url_for("index"))
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
                flash("Account created! Welcome to Mood Journal.", "success")
                return redirect(url_for("index"))
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
# Run locally with `python app.py`
# On Vercel, it imports the `app` object directly (no __main__ block needed)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
