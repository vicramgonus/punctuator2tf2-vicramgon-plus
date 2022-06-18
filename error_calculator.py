# coding: utf-8

"""
Computes and prints the overall classification error and precision, recall, F-score over punctuations.
"""

from numpy import nan
import codecs
import sys
import re
from math import trunc

PUNCTS = [',', '.', ';', ':', '?', '!', '·M']

def tokenize2(input:str, puncts=[',', '.', ';', ':', '?', '!']):
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
      if partial:
        res.append(partial)
         
      res.append(c_i)
      
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

def process_line2(line):
    tokens = tokenize2(line.strip())
    output_tokens = []
    last_is_punct = True
    union = False
    i=0
    while i < (len(tokens)):
        if tokens[i] in PUNCTS:
            if not last_is_punct:
                if  (i > 0) and (tokens[i-1] != ' ') and (i + 1 < len(tokens)) and (tokens[i+1] !=' '):
                  if (output_tokens[-1][-1].isdigit() and tokens[i+1][0].isdigit()) or  (not(output_tokens[-1][-1].isdigit()) and not(tokens[i+1][0].isdigit()) and tokens[i+1] not in PUNCTS) :
                      union = True
                
                else:
                    output_tokens.append(tokens[i])
                    last_is_punct = True
                    union = False

        elif tokens[i][0].isupper():
            first_char = tokens[i][0] if not union else  tokens[i][0].lower()
            rest_chain = '' if len(tokens[i])<2 else tokens[i][1:].lower()
            if union:
                output_tokens[-1] += first_char + rest_chain
            else:
                output_tokens.append(first_char + rest_chain)
            last_is_punct = False
            union = False

        elif tokens[i] != ' ':
            if union:
                output_tokens[-1] += tokens[i].lower()
            else:
                output_tokens.append(tokens[i].lower())
            last_is_punct = False
            union = False
        
        i += 1

    return output_tokens


def compute_error(target_paths, predicted_paths):

    counter = 0

    correct = 0.
    substitutions = 0.
    deletions = 0.
    insertions = 0.

    true_positives = {}
    false_positives = {}
    false_negatives = {}

    for target_path, predicted_path in zip(target_paths, predicted_paths):

        target_punctuation = " "
        predicted_punctuation = " "

        t_i = 0
        p_i = 0

        with open(target_path, 'r') as target, open(predicted_path, 'r') as predicted:

            target_stream = process_line2(re.sub('\s+', ' ', target.readline()).strip())
            predicted_stream = process_line2(re.sub('\s+', ' ', predicted.readline()).strip())

            while target_stream and predicted_stream:
                assert list(map(lambda x: x.lower(), list(filter(lambda x: x not in PUNCTS, target_stream)))) == list(map(lambda x: x.lower(), list(filter(lambda x: x not in PUNCTS, target_stream)))), f"Error in context: {target_stream} vs {predicted_stream}"

                t_i = 0
                p_i = 0
                
                while True:
                    counter += 1
                    target_punctuation = target_stream[t_i]
                    predicted_punctuation = predicted_stream[p_i]
                    
                    if target_punctuation in PUNCTS and predicted_punctuation == target_punctuation:
                        correct += 1
                        true_positives[target_punctuation] = true_positives.get(target_punctuation, 0.) + 1.
                        t_i += 1; p_i += 1
                    
                    elif target_punctuation in PUNCTS and predicted_punctuation in PUNCTS:
                        substitutions += 1
                        false_positives[predicted_punctuation] = false_positives.get(predicted_punctuation, 0.) + 1.
                        false_negatives[target_punctuation] = false_negatives.get(target_punctuation, 0.) + 1.
                        t_i += 1; p_i += 1
                    
                    elif target_punctuation in PUNCTS and predicted_punctuation not in PUNCTS:
                        deletions += 1
                        false_negatives[target_punctuation] = false_negatives.get(target_punctuation, 0.) + 1.
                        t_i += 1
                    
                    elif target_punctuation not in PUNCTS and predicted_punctuation in PUNCTS:
                        insertions += 1
                        false_positives[predicted_punctuation] = false_positives.get(predicted_punctuation, 0.) + 1.
                        p_i += 1

                    elif target_punctuation.lower() == predicted_punctuation.lower():
                        t_i += 1; p_i += 1
                        if target_punctuation.isupper() and predicted_punctuation[0].isupper():
                            correct += 1
                            true_positives['·M'] = true_positives.get('·M', 0.) + 1.
                        elif target_punctuation[0].isupper() and not(predicted_punctuation[0].isupper()):
                          deletions += 1
                          false_negatives['·M'] = false_negatives.get('·M', 0.) + 1.
                        elif not(target_punctuation[0].isupper()) and predicted_punctuation[0].isupper():
                          insertions += 1
                          false_positives['·M'] = false_negatives.get('·M', 0.) + 1.
                        else:
                          correct += 1

                    if t_i >= len(target_stream)-1 and p_i >= len(predicted_stream)-1:
                        break
            
                target_stream = process_line2(re.sub('\s+', ' ', target.readline()).strip())
                predicted_stream = process_line2(re.sub('\s+', ' ', predicted.readline()).strip())

    overall_tp = 0.0
    overall_fp = 0.0
    overall_fn = 0.0

    print("-"*46)
    print("{:<16} {:<9} {:<9} {:<9}".format('PUNCTUATION','PRECISION','RECALL','F-SCORE'))
    for p in PUNCTS:

        overall_tp += true_positives.get(p,0.)
        overall_fp += false_positives.get(p,0.)
        overall_fn += false_negatives.get(p,0.)

        punctuation = p
        precision = (true_positives.get(p,0.) / (true_positives.get(p,0.) + false_positives[p])) if p in false_positives else nan
        recall = (true_positives.get(p,0.) / (true_positives.get(p,0.) + false_negatives[p])) if p in false_negatives else nan
        f_score = (2. * precision * recall / (precision + recall)) if (precision + recall) > 0 else nan        
        print("{:<16} {:<9} {:<9} {:<9}".format(punctuation, round(precision*100,3), round(recall*100,3), round(f_score*100,3)))
    print("-"*46)
    pre = overall_tp/(overall_tp+overall_fp) if overall_fp else nan
    rec = overall_tp/(overall_tp+overall_fn) if overall_fn else nan
    f1 = (2.*pre*rec)/(pre+rec) if (pre + rec) else nan
    print("{:<16} {:<9} {:<9} {:<9}".format("Overall", round(pre*100,3), round(rec*100,3), round(f1*100,3)))
    print("Err: %s%%" % round((100.0 - float(correct) / float(counter-1) * 100.0), 2))
    print("SER: %s%%" % round((substitutions + deletions + insertions) / (correct + substitutions + deletions) * 100, 1))
if __name__ == "__main__":

    if len(sys.argv) > 1:
        target_path = sys.argv[1]
    else:
        sys.exit("Ground truth file path argument missing")

    if len(sys.argv) > 2:
        predicted_path = sys.argv[2]
    else:
        sys.exit("Model predictions file path argument missing")

    compute_error([target_path], [predicted_path])    
        