import classify

if __name__ == '__main__' :
   print ("starting")
   classify.config("efficientnetv2-s")
   result = classify.classify ("picture.jpg")
   print (result)
   print ("done")
   
