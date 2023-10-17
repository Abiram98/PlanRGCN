from feature_extraction.sparql import Endpoint


endpoint = Endpoint("http://130.225.39.154:8891/sparql")
predicate = "http://www.wikidata.org/prop/qualifier/P2308"
query = f"""
SELECT * {{
   ?s <{predicate}> ?o 
}} LIMIT 5
"""
res = endpoint.run_query(query)
print(res)
