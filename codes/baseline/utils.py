import numpy as np
import shutil
import os
import time 
import sys
from matplotlib import pyplot as plt 
import seaborn as sns
import cv2
from pathlib import Path



def prepareDirs(config):
	print(config.output_dir)
	if not os.path.exists(config.output_dir):
		os.makedirs(config.output_dir)


	if not os.path.exists(Path(str(config.output_dir)+"/saved_models")):
		os.makedirs(Path(str(config.output_dir)+"/saved_models"))


	
	if not os.path.exists(config.save_model_dir):
		os.makedirs(config.save_model_dir)

	


def prepareData(data):
	finalData=[]
	for i in data:
		for j in i:
			finalData.append(j)

	finalData=np.array(finalData)
	return finalData








			 


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
	global last_time, begin_time
	if current == 0:
		begin_time = time.time()  # Reset for new bar.

	cur_len = int(TOTAL_BAR_LENGTH*current/total)
	rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

	sys.stdout.write(' [')
	for i in range(cur_len):
		sys.stdout.write('=')
	sys.stdout.write('>')
	for i in range(rest_len):
		sys.stdout.write('.')
	sys.stdout.write(']')

	cur_time = time.time()
	step_time = cur_time - last_time
	last_time = cur_time
	tot_time = cur_time - begin_time

	L = []
	L.append('  Step: %s' % format_time(step_time))
	L.append(' | Tot: %s' % format_time(tot_time))
	if msg:
		L.append(' | ' + msg)

	msg = ''.join(L)
	sys.stdout.write(msg)
	for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
		sys.stdout.write(' ')

	# Go back to the center of the bar.
	for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
		sys.stdout.write('\b')
	sys.stdout.write(' %d/%d ' % (current+1, total))

	if current < total-1:
		sys.stdout.write('\r')
	else:
		sys.stdout.write('\n')
	sys.stdout.flush()


def format_time(seconds):
	days = int(seconds / 3600/24)
	seconds = seconds - days*3600*24
	hours = int(seconds / 3600)
	seconds = seconds - hours*3600
	minutes = int(seconds / 60)
	seconds = seconds - minutes*60
	secondsf = int(seconds)
	seconds = seconds - secondsf
	millis = int(seconds*1000)

	f = ''
	i = 1
	if days > 0:
		f += str(days) + 'D'
		i += 1
	if hours > 0 and i <= 2:
		f += str(hours) + 'h'
		i += 1
	if minutes > 0 and i <= 2:
		f += str(minutes) + 'm'
		i += 1
	if secondsf > 0 and i <= 2:
		f += str(secondsf) + 's'
		i += 1
	if millis > 0 and i <= 2:
		f += str(millis) + 'ms'
		i += 1
	if f == '':
		f = '0ms'
	return f



def compareAlgorithms(predictAlgoirthm1, predictAlgoirthm2, trueValues):

	tpCount=0
	tnCount=0
	fpCount=0
	fnCount=0
	
	print(len(predictAlgoirthm1))
	print(len(predictAlgoirthm2))	

	# count = 0
	for currentAlgo1,currentAlgo2,currentTrueValues in zip(predictAlgoirthm1, predictAlgoirthm2, trueValues):
		
		if(currentAlgo1 == 0 and currentAlgo2 == 0 and currentTrueValues == 0):
			tpCount += 1
		
		elif(currentAlgo1 == 1 and currentAlgo2 == 1 and currentTrueValues == 1):
			tnCount += 1


		elif(currentAlgo1 == 0 and currentAlgo2 == 1 and currentTrueValues == 1):
			fpCount += 1
		
		elif(currentAlgo1 == 1 and currentAlgo2 == 0 and currentTrueValues == 1):
			fpCount += 1
		
		elif(currentAlgo1 == 0 and currentAlgo2 == 0 and currentTrueValues == 1):
			fpCount += 1
		


		elif(currentAlgo1 == 0 and currentAlgo2 == 1 and currentTrueValues == 0):
			fnCount += 1
		
		elif(currentAlgo1 == 1 and currentAlgo2 == 0 and currentTrueValues == 0):
			fnCount += 1

		elif(currentAlgo1 == 1 and currentAlgo2 == 1 and currentTrueValues == 0):
			fnCount += 1
		
	


	
	totalCount = len(trueValues)
	
	print(totalCount)
	temp = tpCount + tnCount + fnCount + fpCount
	print(temp)	
	print("***********VALUES***********")
	print("TP: ",tpCount/totalCount)
	print("TN: ",tnCount/totalCount)
	print("FP: ",fpCount/totalCount)
	print("FN: ",fnCount/totalCount)
	


