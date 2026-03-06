from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from Interpreter.ChaiScript import run_web
from Interpreter.ast_serializer import ast_to_json

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
    
