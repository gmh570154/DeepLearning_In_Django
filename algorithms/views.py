from django.shortcuts import render
from .models import HistoryInfo
from django.core import serializers
from django.core.serializers import serialize
from django.http import HttpResponse, JsonResponse
import json


def add_view_log(request):  #  调用接口记录访问日志
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR', '')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]  # 这里是真实的ip
    else:
        ip = request.META.get('REMOTE_ADDR')  # 这里获得代理ip
        HistoryInfo.objects.create(hostname=ip)

    return render(request, 'index.html', {
        'hostname': ip,
        'count': HistoryInfo.objects.count()
    })

def get_all_view_logs(request):  # 获取所有的访问日志的记录，返回json数据格式
    result = HistoryInfo.objects.all()
    json_data = serialize("json", result, ensure_ascii=False)
    return HttpResponse(json_data, content_type='application/json; charset=utf-8')


def get_all_view_logs1(request):  # 获取所有的访问日志的记录，返回json数据格式
    result = {"password": "password", "encrypt": "encrpt"}
    return HttpResponse(json.dumps(result), content_type="application/json")


def get_all_view_logs2(request):  # 获取所有的访问日志的记录，返回json数据格式
    result = HistoryInfo.objects.all()
    #json_data = serialize("json", result, ensure_ascii=False)
    #return HttpResponse(json_data, content_type='application/json; charset=utf-8')
    #return HttpResponse(json.dumps(result, ensure_ascii=False), content_type="application/json,charset=utf-8")
    data = {}
    data["return"] = json.loads(serializers.serialize("json", result, ensure_ascii=False))
    return JsonResponse(data, content_type='application/json; charset=utf-8')

