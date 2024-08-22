import subprocess, os
import json
import sys
DISABLE_WARNING=True
try:
    PATH_JAR = os.environ["QG_JAR"] #'/qp/target/qp-1.0-SNAPSHOT.jar'
except KeyError as e:
    print(e)
    print('Please provide path to jar file for QP construction: "QG_JAR"')
    sys.exit(-1)

import jpype
import jpype.imports
from jpype.types import *
jpype.startJVM(classpath=[PATH_JAR])
from com.org import App

if DISABLE_WARNING:
    App.disableWarns()
def get_query_graph(query):
    return json.loads(str(App.getQueryGraph(query)))

    return subprocess.run(['java', '-jar', PATH_JAR, 'qg', query], stdout=subprocess.PIPE).stdout.decode('utf-8')
    return json.loads(subprocess.run(['java', '-jar',PATH_JAR, 'qg', query], stdout=subprocess.PIPE).stdout.decode('utf-8'))