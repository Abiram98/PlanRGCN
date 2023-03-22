from SPARQLWrapper import SPARQLWrapper, JSON

class Query:
    def __init__(self, endpoint_url):
        pass
        self.sparql = SPARQLWrapper(endpoint_url)
        self.sparql.setReturnFormat(JSON)
        
    def run_query(self, query_str: str):
        self.query_str = query_str
        try:
            self.sparql.setQuery(query_str)
        except Exception:
            print("Query could not be executed!")
            exit()
        self.results = self.sparql.query().convert()
        return self.results

