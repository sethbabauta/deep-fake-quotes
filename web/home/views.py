from django.shortcuts import render

# Views for the basic web app go here
from django.http import HttpResponse

def index(req):
	return HttpResponse("Hello world. I am your master.")