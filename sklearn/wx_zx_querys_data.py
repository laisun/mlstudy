#!/usr/bin/env python
#-*- coding:utf-8 -*-

import os,sys
import csv

from sklearn import preprocessing
from sklearn import metrics
import numpy as np
import collections
from sk_data_reprecessing import DataPreprocessing

cates = []
#numbers = "cnt_1,cnt_2,cnt_3,cnt_4,cnt_5,cnt_6,cnt_7,cnt_8,cnt_9,cnt_10,cnt_11,cnt_12,cnt_13,cnt_14,cnt_15,cnt_16,cnt_17,cnt_18,cnt_19,cnt_20,cnt_21,cnt_22,cnt_23,cnt_24,cnt_25,cnt_26,cnt_27,cnt_28,cnt_29,p".split(",")
#numbers = ["p"]
numbers = "cnt_1,cnt_2,cnt_3,cnt_4,cnt_5,cnt_6,cnt_7,cnt_8,cnt_9,cnt_10,cnt_11,cnt_12,cnt_13,cnt_14,cnt_15,cnt_16,cnt_17,cnt_18,cnt_19,cnt_20,cnt_21,cnt_22,cnt_23,cnt_24,cnt_25,cnt_26,cnt_27,cnt_28,cnt_29,cnt_30,cnt_31,cnt_32,cnt_33,cnt_34,cnt_35,cnt_36,cnt_37,cnt_38,cnt_39,cnt_40,cnt_41,cnt_42,cnt_43,cnt_44,cnt_45,cnt_46,cnt_47,cnt_48,cnt_49,cnt_50,cnt_51,cnt_52,cnt_53,cnt_54,cnt_55,cnt_56,cnt_57,cnt_58,cnt_59,cnt_60,cnt_61,cnt_62,cnt_63,cnt_64,cnt_65,cnt_66,cnt_67,cnt_68,cnt_69,cnt_70,cnt_71,cnt_72,cnt_73,cnt_74,cnt_75,cnt_76,cnt_77,cnt_78,cnt_79,cnt_80,cnt_81,cnt_82,cnt_83,cnt_84,cnt_85,cnt_86,cnt_87,cnt_88,cnt_89,cnt_90,cnt_91,cnt_92,cnt_93,cnt_94,cnt_95,cnt_96,cnt_97,cnt_98,cnt_99,cnt_100,cnt_101,cnt_102,cnt_103,cnt_104,cnt_105,cnt_106,cnt_107,cnt_108,cnt_109,cnt_110,cnt_111,cnt_112,cnt_113,cnt_114,cnt_115,cnt_116,cnt_117,cnt_118,cnt_119,cnt_120,cnt_121,cnt_122,cnt_123,cnt_124,cnt_125,cnt_126,cnt_127,cnt_128,cnt_129,cnt_130,cnt_131,cnt_132,cnt_133,cnt_134,cnt_135,cnt_136,cnt_137,cnt_138,cnt_139,cnt_140,cnt_141,cnt_142,cnt_143,cnt_144,cnt_145,cnt_146,cnt_147,cnt_148,cnt_149,cnt_150,cnt_151,cnt_152,cnt_153,cnt_154,cnt_155,cnt_156,cnt_157,cnt_158,cnt_159,cnt_160,cnt_161,cnt_162,cnt_163,cnt_164,cnt_165,cnt_166,cnt_167,cnt_168,cnt_169,cnt_170,cnt_171,p".split(",")

y_column = 'y'
v_to_idx = { 
             '0':0,
	         '1':1,
}

adult_data_processing = DataPreprocessing(
                cates,
			    numbers,
			    y_column,
                v_to_idx
)

path='/home/lx/work/mmdataapp/wxalonso/wxqueryanlysis/data/zx_user_train/'
path += 'alonsoli_zx_game_querys_201704_05_06_unions_gbie_with_intest_p.csv.topics_2.csv'

adult_data_processing.catefeat_v_to_id_mapping( path)

def read_adult_data():
  y_train,X_train,y_test,X_test = \
      adult_data_processing.read_train_test(
           path + '_train.txt',
	       path + '_test.txt'
      )	
  return y_train,X_train,y_test,X_test

