import sys

# Args
arg1 = sys.argv[1] # Path
arg2 = sys.argv[2] # Config
arg3 = sys.argv[3] # Dbg

# Open
f = open(arg1,'r')
filedata = f.read()
f.close()

# Config
newdata = filedata.replace("{{c}}", arg2)

# Dbg
if arg3 == True:
    newdata = newdata.replace("{{d}}", "//")
else:
    newdata = newdata.replace("{{d}}", "")

# Write and close
f = open(arg1,'w')
f.write(newdata)
f.close()