from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from Interpreter.ChaiScript import run_web
from Interpreter.ast_serializer import ast_to_json
from .models import SharedCode

@csrf_exempt
def run_chai(request):

    if request.method == "POST":
        code = request.POST.get("code", "")

        output, trace, ast = run_web(code)
        ast_json = ast_to_json(ast)

        return JsonResponse({
            "output": output,
            "trace": trace,
            "ast": ast_json
        })

    return JsonResponse({
        "error": "POST request required"
    }, status=400)
    

@csrf_exempt
def share_code(request):
    if request.method == "POST":
        code = request.POST.get("code", "")
        obj = SharedCode.objects.create(code=code)
        return JsonResponse({"share_id": obj.share_id})
    return JsonResponse({"error": "POST required"}, status=400)

def load_shared(request, share_id):
    try:
        obj = SharedCode.objects.get(share_id=share_id)
        # If JS is fetching JSON, return JSON
        if request.headers.get("Accept", "").startswith("application/json") or request.GET.get("json"):
            return JsonResponse({"code": obj.code})
        # Otherwise render the IDE page — JS will load the code via fetch
        return render(request, "engine/index.html")
    except SharedCode.DoesNotExist:
        return JsonResponse({"error": "Not found"}, status=404)
