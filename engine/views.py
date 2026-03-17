from django.shortcuts import render, get_object_or_404, redirect
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.contrib.auth.models import User
from django.db.models import Count, Sum, F
import time, json

from Interpreter.ChaiScript import run_web
from Interpreter.ast_serializer import ast_to_json
from .models import SharedCode, Project, VersionSnapshot, Challenge, ChallengeSubmission, UserProfile, HomeComment


# ── helpers ────────────────────────────────────────────────
def _get_avatar(user):
    try:
        sa = user.socialaccount_set.filter(provider='google').first()
        if sa:
            return sa.extra_data.get('picture', '')
    except Exception:
        pass
    return ''


# ── existing: run code ─────────────────────────────────────
@csrf_exempt
def run_chai(request):
    if request.method == "POST":
        try:
            code = json.loads(request.body).get("code", "")
        except Exception:
            code = request.POST.get("code", "")
        # increment run counter if logged in
        if request.user.is_authenticated:
            UserProfile.objects.filter(user=request.user).update(
                total_runs=F('total_runs') + 1
            )
        output, trace, ast = run_web(code)
        ast_json = ast_to_json(ast)
        return JsonResponse({"output": output, "trace": trace, "ast": ast_json})
    return JsonResponse({"error": "POST request required"}, status=400)


# ── existing: share ────────────────────────────────────────
@csrf_exempt
def share_code(request):
    if request.method == "POST":
        try:
            code = request.POST.get("code", "")
            obj = SharedCode.objects.create(code=code)
            return JsonResponse({"share_id": obj.share_id})
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    return JsonResponse({"error": "POST required"}, status=400)


def load_shared(request, share_id):
    try:
        obj = SharedCode.objects.get(share_id=share_id)
        if request.headers.get("Accept", "").startswith("application/json") or request.GET.get("json"):
            return JsonResponse({"code": obj.code})
        return render(request, "engine/index.html")
    except SharedCode.DoesNotExist:
        return JsonResponse({"error": "Not found"}, status=404)


def docs(request):
    return render(request, "engine/docs.html")


# After
def home(request):
    if request.method == "POST":
        body = request.POST.get("body", "").strip()
        if body:
            HomeComment.objects.create(
                user=request.user if request.user.is_authenticated else None,
                body=body
            )
        return redirect('home')

    comments = HomeComment.objects.select_related('user').order_by('-created_at')[:5]

    # attach avatar to each comment
    comment_data = []
    for c in comments:
        avatar = ''
        if c.user:
            try:
                sa = c.user.socialaccount_set.filter(provider='google').first()
                if sa:
                    avatar = sa.extra_data.get('picture', '')
            except Exception:
                pass
        comment_data.append({'comment': c, 'avatar': avatar})

    return render(request, "engine/home.html", {
        "comment_data": comment_data,
    })


def custom_404(request, exception):
    return render(request, "engine/404.html", status=404)


# ── projects: save ─────────────────────────────────────────
@csrf_exempt
@login_required
def save_project(request):
    if request.method == "POST":
        data       = json.loads(request.body)
        code       = data.get("code", "")
        title      = data.get("title", "Untitled")
        project_id = data.get("project_id")

        if project_id:
            proj = get_object_or_404(Project, id=project_id, user=request.user)
            # snapshot before overwriting
            VersionSnapshot.objects.create(project=proj, code=proj.code, label='auto')
            # keep only last 20 snapshots
            old = proj.snapshots.values_list('id', flat=True)[20:]
            VersionSnapshot.objects.filter(id__in=list(old)).delete()
            proj.code  = code
            proj.title = title
            proj.save()
        else:
            proj = Project.objects.create(user=request.user, title=title, code=code)

        return JsonResponse({"project_id": proj.id, "title": proj.title})
    return JsonResponse({"error": "POST required"}, status=400)


# ── projects: list ─────────────────────────────────────────
@login_required
def list_projects(request):
    projects = request.user.projects.values('id', 'title', 'updated_at')
    return JsonResponse({"projects": list(projects)}, json_dumps_params={"default": str})


# ── projects: load ─────────────────────────────────────────
@login_required
def load_project(request, project_id):
    proj = get_object_or_404(Project, id=project_id, user=request.user)
    snapshots = list(proj.snapshots.values('id', 'label', 'created_at')[:10])
    return JsonResponse({
        "project_id": proj.id,
        "title": proj.title,
        "code": proj.code,
        "snapshots": snapshots,
    }, json_dumps_params={"default": str})


# ── version history: restore ───────────────────────────────
@login_required
def restore_snapshot(request, snapshot_id):
    snap = get_object_or_404(VersionSnapshot, id=snapshot_id, project__user=request.user)
    return JsonResponse({"code": snap.code, "label": snap.label})


# ── profile page ───────────────────────────────────────────
@login_required
def profile(request):
    user    = request.user
    profile = getattr(user, 'profile', None)
    if not profile:
        profile = UserProfile.objects.create(user=user)

    projects   = user.projects.all()[:10]
    solved     = user.submissions.filter(passed=True).values('challenge').distinct().count()
    avatar     = profile.avatar_url or _get_avatar(user)

    return render(request, "engine/profile.html", {
        "profile":  profile,
        "projects": projects,
        "solved":   solved,
        "avatar":   avatar,
    })


# ── leaderboard ────────────────────────────────────────────
def leaderboard(request):
    # top by total runs
    top_runners = UserProfile.objects.select_related('user').order_by('-total_runs')[:20]
    # top by challenges solved
    top_solvers = (
        ChallengeSubmission.objects
        .filter(passed=True)
        .values('user__username')
        .annotate(solved=Count('challenge', distinct=True))
        .order_by('-solved')[:20]
    )
    return render(request, "engine/leaderboard.html", {
        "top_runners": top_runners,
        "top_solvers": top_solvers,
    })


# ── challenges: list ───────────────────────────────────────
def challenge_list(request):
    challenges = Challenge.objects.filter(is_active=True)
    solved_ids = set()
    if request.user.is_authenticated:
        solved_ids = set(
            request.user.submissions.filter(passed=True)
            .values_list('challenge_id', flat=True)
        )
    return render(request, "engine/challenges.html", {
        "challenges": challenges,
        "solved_ids": solved_ids,
    })


# ── challenges: detail + submit ────────────────────────────
def challenge_detail(request, slug):
    challenge = get_object_or_404(Challenge, slug=slug, is_active=True)
    best = None
    if request.user.is_authenticated:
        best = (
            request.user.submissions
            .filter(challenge=challenge, passed=True)
            .order_by('exec_time').first()
        )
    return render(request, "engine/challenge_detail.html", {
        "challenge": challenge,
        "best": best,
    })


@csrf_exempt
@login_required
def submit_challenge(request, slug):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=400)

    challenge = get_object_or_404(Challenge, slug=slug, is_active=True)
    data      = json.loads(request.body)
    user_code = data.get("code", "")

    # Run ONLY the user's code
    start = time.perf_counter()
    output, trace, _ = run_web(user_code)
    elapsed = round(time.perf_counter() - start, 4)

    # Compare output in Python — test_code field = expected output lines
    output_lines   = [l.strip() for l in output.strip().splitlines() if l.strip()]
    expected_lines = [l.strip() for l in challenge.test_code.strip().splitlines() if l.strip()]
    passed = (output_lines == expected_lines)

    ChallengeSubmission.objects.create(
        user=request.user,
        challenge=challenge,
        code=user_code,
        passed=passed,
        output=output,
        exec_time=elapsed,
    )

    return JsonResponse({
        "passed":    passed,
        "output":    output,
        "exec_time": elapsed,
    })