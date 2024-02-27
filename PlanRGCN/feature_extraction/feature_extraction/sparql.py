from SPARQLWrapper import SPARQLWrapper, JSON, POST

class Endpoint:
    def __init__(self, endpoint_url):
        self.sparql = SPARQLWrapper(endpoint_url, defaultGraph='http://localhost:8890/dataspace')
        self.sparql.setReturnFormat(JSON)
        self.sparql.setMethod(POST)
    def run_query(self, query_str: str):
        if not hasattr(self,'sparql'):
            print('SPARQL Endpoint has not been initialised!!!')
            exit()
        try:
            self.sparql.setQuery(query_str)
        except Exception:
            print("Query could not be executed!")
            return query_str
        results = self.sparql.query().convert()
        return results