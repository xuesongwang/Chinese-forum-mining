# -*- coding:utf-8 -*-
__author__ = 'Zonglin Di'
import logging
import getpass
import sys


''' Define MyLog Class for logging'''
class MyLog(object):

	def __init__(self): 
		self.user = getpass.getuser()
		self.logger = logging.getLogger(self.user)
		self.logger.setLevel(logging.INFO)
		logFile = 'getCommentInfo'+'.log' #Log File Name
		formatter = logging.Formatter('%(asctime)-12s %(levelname)-8s %(name)-10s %(message)-12s')

		'''Print on the Console and log into the file'''
		logHand = logging.FileHandler(logFile)
		logHand.setFormatter(formatter)
		logHand.setLevel(logging.INFO) #Determine which level will be recorded

		logHandSt = logging.StreamHandler()
		logHandSt.setFormatter(formatter)

		self.logger.addHandler(logHand)
		self.logger.addHandler(logHandSt)

	'''5 levels for logging'''
	def debug(self,msg):
		self.logger.debug(msg)

	def info(self,msg):
		self.logger.info(msg)

	def warn(self,msg):
		self.logger.warn(msg)

	def error(self,msg):
		self.logger.error(msg)

	def critical(self,msg):
		self.logger.critical(msg)

if __name__ == '__main__':
	mylog = MyLog()
	mylog.debug("I'm debug")
	mylog.info("I'm info")
	mylog.warn("I'm warn")
	mylog.error("I'm error")
	mylog.critical("I'm critical")
