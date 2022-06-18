# coding=utf-8
# Copyright 2016 Ottokar Tilk and Tanel Alumäe
# The following code is available on:
# https://github.com/ottokart/punctuator2/blob/master/example/dont_run_me_run_the_other_script_instead.py
# In it, functions are defined to exchange all punctuation marks in a dataset for 
# descriptors of said punctuations.

from __future__ import division, print_function
from nltk.tokenize import word_tokenize

import nltk
import os
from io import open
import re
import sys

nltk.download("punkt")

NUM = "<NUM>"
START = "<START>"
END = "<END>"
BREAK = "|BREAK"

EOS_PUNCTS = {".": ".PERIOD", "?": "?QUESTIONMARK", "!": "!EXCLAMATIONMARK"}
INS_PUNCTS = {",": ",COMMA", ";": ";SEMICOLON", ":": ":COLON"}

forbidden_symbols = re.compile(r"[\[\]\(\)\/\\\>\<\=\+\_\*]")
multiple_punct = re.compile(r"([\.\?\!\,\:\;])(?:[\.\?\!\,\:\;]){1,}")

def tokenize(input:str, puncts=[',', '.', ';', ':', '?', '!']):
  # Inicializamos una lista vacía que contendrá, en cada paso, los tokens
  # completos procesados hasta el momento.
  res = []

  # Se inicializa una cadena vacía que contendrá, en cada paso, la secuencia
  # parcial de caractéres de la cadena del token en procesamiento. 
  partial = ''

  # Reading type
  cur_read_type = None
  
  # En cada paso (hasta terminar de procesar todos los caracteres de la cadena)
  for i in range(len(input)):
    
    # Se lee el carácter i-ésimo de la cadena
    c_i = input[i] 
    
    # Si dicho carácter corresponde a un espacio o a un signo de puntuación
    # entonces, el token que estábamos procesando finaliza, luego
    if c_i in [' '] + puncts:
      
      # Ha de añadirse (junto al signo de corte), como nuevos tokens.
      # Sólo se añadirá no es vacío ( si no corresponde al espacio, resp.). 
      res += list(filter(lambda x: x.strip() != '', [partial, c_i]))
      
      # Y reseteamos la cadena parcial de token a vacío
      partial = ''
      cur_read_type = None
    
    # En otro caso, 
    else:
      # Añadimos el carácter a la cadena parcial del token procesamiento
      # si los tipos no coinciden entendemos que corresponden a cadenas distintas

      if (c_i.isdigit()):
          if(cur_read_type == 'string'):
            res += [partial]
            partial = ''
              
          cur_read_type = 'number'
      
      else:
          if(cur_read_type == 'number'):
            res += [partial]
            partial = ''
   
          cur_read_type = 'string'
      
      
      partial += c_i
    
    #Y continúa el procesamiento del siguiente carácter.
  
  # El token leído en última instancia es añadido a la lista de tokens (siempre
  # que no sea vacío)
  res += [partial] if partial.strip() != '' else []

  # Finalmente, se devuelve la lista de tokens
  return res

def process_line(line):

    tokens =  tokenize(line.strip())
    output_tokens = []
    prev_token = 'null'

    for token in tokens:
        if token in INS_PUNCTS:
            if  prev_token not in INS_PUNCTS and prev_token not in EOS_PUNCTS:
                output_tokens.append(INS_PUNCTS[token])
        elif token in EOS_PUNCTS:
            if  prev_token not in INS_PUNCTS and prev_token not in EOS_PUNCTS:
                output_tokens.append(EOS_PUNCTS[token])
        elif re.fullmatch('\d+', token):
            output_tokens.append(NUM)

        elif token[0].isupper():
            if prev_token in EOS_PUNCTS or prev_token in INS_PUNCTS:
              output_tokens[-1] += 'M'
              output_tokens.append(token.lower())
            else:
              output_tokens.append('·M')
              output_tokens.append(token.lower())
        else:
            output_tokens.append(token.lower())
        
        prev_token = token

    return " ".join([START] + output_tokens + [END, BREAK])

