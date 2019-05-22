#-*- coding: utf-8 -*-
from django.shortcuts import render
from util import mongodbHelper
from django.http import HttpResponse
from bson.json_util import dumps


def get_ganji_all(request):  # 获取赶集网爬虫的所有数据
    db = mongodbHelper.mongoDB()
    data = db.find_all("ganjicol")
    return HttpResponse(dumps(data,ensure_ascii=False),content_type="application/json,charset=utf-8")