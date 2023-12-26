from feature_extraction.predicates.pred_util import *
from unicodedata import normalize
import rdflib.plugins.sparql.parser as SPARQLparser

class LiteralFreqExtractor(ExtractorBase):
    def __init__(
        self,
        endpoint: Endpoint,
        output_dir: str,
        literal_file="literals.json",
        batch_size = 1000
    ) -> None:
        super().__init__(endpoint, output_dir, literal_file)
        self.batch_size = batch_size
        self.literal_file = literal_file
        if os.path.exists(literal_file):
            literals = json.load(open(literal_file, "r"))
            self.literals = [x['o']['value'] for x in literals['results']['bindings'] ]
            self.literal_types = [x['o']['type'] for x in literals['results']['bindings'] ]
            # for backward compatibility (load_batches)
            self.predicates = self.literals
        
        self.batch_output_dir = f"{output_dir}/literals_stat/batches"
        os.system(f"mkdir -p {self.batch_output_dir}")
        # the
        # path to where the responses are saved with the features.
        self.batch_output_response_dir = f"{output_dir}/literals_stat/batches_response_stats"
        os.system(f"mkdir -p {self.batch_output_response_dir}")
        os.system(f"mkdir -p {self.batch_output_response_dir}/freq")
        os.system(f"mkdir -p {self.batch_output_response_dir}/pred_lits")

    def query_distinct_lits(self):
        query = LiteralStatQueries.extract_all_literals()
        res = self.endpoint.run_query(query)
        res_fp = f"{self.batch_output_response_dir}/{self.literal_file}"
        json.dump(res, open(res_fp, "w"))
        print(f"batch literals extracted!")
        
    def query_batches(self, batch_start=1, batch_end=2):
        if not hasattr(self, "batches"):
            self.load_batches()
        if batch_end == -1:
            batch_end = len(self.batches)
        save_path = self.batch_output_response_dir
        os.system(f"mkdir -p {save_path}")
        print(f"Literals Stats are saved to: {save_path}")
        print(f"Beginning extraction of batch {batch_start - 1} to {batch_end - 1}")
        for i, b in enumerate(self.batches[batch_start - 1 : batch_end - 1]):
            for query_generator, name in zip(
                [
                    LiteralStatQueries.freq_lits,
                ],
                ["freq"],
            ):
                query = query_generator(b)
                try:
                    res_fp = f"{save_path}/{name}/batch_{batch_start+i}.json"
                    if os.path.exists(res_fp):
                        continue
                    res = self.endpoint.run_query(query)
                    json.dump(res, open(res_fp, "w"))
                    print(f"batch {batch_start+i}/{len(self.batches)} extracted!")
                except Exception:
                    print(f"Did not work for {batch_start+i}")
                    with open("/data/unprocessed_batches3.log","a") as f:
                        f.write(query)
                        f.write("\n\n\n\n")
        print("exiting after freq")
        exit()
        for i, b in enumerate(self.batches[batch_start - 1 : batch_end - 1]):
            for query_generator, name in zip(
                [
                    LiteralStatQueries.pred_lits,
                ],
                [ "pred_lits"],
            ):
                query = query_generator(b)
                res = self.endpoint.run_query(query)
                
                res_fp = f"{save_path}/{name}/batch_{batch_start+i}.json"
                json.dump(res, open(res_fp, "w"))
                print(f"batch {batch_start+i}/{len(self.batches)} extracted!")


class LiteralStatQueries:
    def extract_all_literals():
        return """
    SELECT distinct( ?o) {
        ?s ?p ?o.
        FILTER isLiteral(?o)
    }
    """
    def pred_str_gen(batch):
        pred_str = ""
        batch = LiteralStatQueries.retrieve_good_literals(batch)
        for p in batch:
            """if '"' in p[0]:
                p[0] = p[0].replace('"', '\\"').replace('\\\\', '\\')"""
            if ("Point" in p[0]):
                continue
            if p[1] == 'typed-literal':
                pred_str += f"({p[0]})"
            else:
                pred_str += f"(\"{p[0]}\")"
        return pred_str

    def retrieve_good_literals(batch, error_file = "/data/bad_literals.txt"):
        good_lits = []
        illegal_terms = []
        legal_terms = []
        for i in batch:
            if i[1] == "literal":
                query1 = f"""SELECT * WHERE {{ ?s ?p \"{i[0]}\"}}"""
            else:
                query1 = f"""SELECT * WHERE {{ ?s ?p {i[0]}}}"""
            try:
                SPARQLparser.parseQuery(query1)
            except Exception as e:
                illegal_terms.append(i)
            else:
                legal_terms.append(i)
        
        with open(error_file, 'a') as f:
            f.write(str(illegal_terms))
        
        return legal_terms
                

    def freq_lits(batch):
        ent_str = LiteralStatQueries.pred_str_gen(batch)
        return f"""SELECT ?e (COUNT( *) AS ?entities) WHERE {{
            VALUES (?e) {{ {ent_str}}}
        ?s ?p2 ?e .
        }}
        GROUP BY ?e
        """

    def pred_lits(batch):
        ent_str = LiteralStatQueries.pred_str_gen(batch)
        """This returns the count of unique entities in both subject and object positions."""
        return f"""SELECT ?e (COUNT( ?p2) AS ?entities) WHERE {{
            VALUES (?e) {{ {ent_str}}}
            ?s ?p2 ?e .
        }}
        GROUP BY ?e
        """


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Literal Stat Extractor",
        description="Extract Literal stats from SPARQL Endpoint",
        epilog="Text at the bottom of help",
    )

    parser.add_argument("task")
    parser.add_argument("-e", "--endpoint")
    parser.add_argument("--dir", "--output_dir")
    parser.add_argument("--lits_file")
    
    parser.add_argument("--batch_start")
    parser.add_argument("--batch_end")

    args = parser.parse_args()

    if args.task == "distinct-literals":
        output_dir = f"{args.dir}"
        os.system(f"mkdir -p {output_dir}")
        endpoint = Endpoint(args.endpoint)
        os.system(f"mkdir -p {args.dir}")
        extractor = LiteralFreqExtractor(endpoint, output_dir, args.lits_file)
        extractor.query_distinct_lits()
        
    if args.task == "extract-lits-stat":
        output_dir = f"{args.dir}"
        os.system(f"mkdir -p {output_dir}")
        endpoint = Endpoint(args.endpoint)
        os.system(f"mkdir -p {args.dir}")
        extractor = LiteralFreqExtractor(endpoint, output_dir, args.lits_file, batch_size=200)
        extractor.load_batches()
        extractor.query_batches(int(args.batch_start), int(args.batch_end))
        
        #python3 -m feature_extraction.literal_utils -e http://172.21.233/arql/ --dir /data/extracted_features_dbpedia2016 --lits_file literals.json --batch_start -1 --batch_end -1 distinct-literals
        #python3 -m feature_extraction.literal_utils -e httpi//172.21.233.14:8892/sparql/ --dir /data/extracted_features_dbpedia2016 --lits_file /data/extracted_features_dbpedia2016/literals_stat/batches_response_stats/literals.json --batch_start 1 --batch_end -1 extract-lits-stat

#python3 -m feature_extraction.literal_utils -e http://130.225.39.154:8892/sparql/ --dir /data/planrgcn_features/extracted_features_dbpedia2016 --lits_file /data/planrgcn_features/extracted_features_dbpedia2016/literals_stat/batches_response_stats/literals.json --batch_start 1 --batch_end -1 extract-lits-stat
