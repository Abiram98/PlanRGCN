from graph_construction.stack import Stack


class QueryPlanUtils:
    "filter rel definied in getjointype method"

    def get_relations(op):
        match op:
            case "conditional":
                return 11
            case "leftjoin":
                return 12
            case "join":
                return 13
            case "union":
                return 14
            case "minus":
                return 15
        raise Exception("Operation undefind " + op)

    def extract_triples(data: dict):
        triple_data = []
        stack = Stack()
        stack.push(data)
        while not stack.is_empty():
            current = stack.pop()
            if "subOp" in current.keys():
                for node in reversed(current["subOp"]):
                    stack.push(node)
            if current["opName"] == "Triple":
                triple_data.append(current)
        return triple_data

    def extract_triples_filter(data: dict):
        triple_data = []
        stack = Stack()
        stack.push(data)
        while not stack.is_empty():
            current = stack.pop()
            if "subOp" in current.keys():
                for node in reversed(current["subOp"]):
                    stack.push(node)
            if current["opName"] == "Triple":
                triple_data.append(current)
        return triple_data

    def map_extracted_triples(triple_dct: list[dict], trpl_list: list):
        res_t = list()
        for t in trpl_list:
            if t in triple_dct:
                res_t.append(t)
        return res_t
