import sys
import os

def check_on_line(arr):
  saw_0 = False
  consistency_error = False
  for e in arr:
    if not saw_0 and e == 0:
      saw_0 = True
    if e == 1 and saw_0:
      print('0 And Then 1')
      consistency_error = True
  return consistency_error
  
def parse_line(line):
  arr = [int(e) for e in line.split(',')[1:-1]]
  return arr

def main():
  error = False
  with open('log.txt', 'r') as logFile:
    for line in logFile:
      arr = parse_line(line)
      error = error or check_on_line(arr)
  print('Error')

main()