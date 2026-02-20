from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from Interpreter.ChaiScript import run_web

@csrf_exempt
def run_chai(request):
    if request.method == "POST":
        code = request.POST.get("code", "")
        output = run_web(code)
        return JsonResponse({"output": output})
    


