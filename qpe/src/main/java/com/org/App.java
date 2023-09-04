package  com.org;

import com.org.Algebra.PredicateLogExtractor;
import com.org.Algebra.Utils;

public class App
{
    public static void main( String[] args )
    {
        String task = args[0];
        switch (task){
            case "test": {
                test();
                break;
            }
            case "extract-query-plans":{
                break;
            }
            case "extract-predicates-query-log":{
                PredicateLogExtractor.run(args[1], args[2]);
                break;
            }
        }
        
    }

    public static void test(){
        String query = "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> PREFIX dbpr: <http://dbpedia.org/resource/> SELECT * WHERE { dbpr:The_Troop rdf:type ?obj }";
        query = """
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            PREFIX wd: <http://www.wikidata.org/entity/>
            SELECT (COUNT(DISTINCT (?occName))AS ?num_works) ?presName

            WHERE {
            
            ?president wdt:P39/wdt:P279* wd:Q248577 .
            ?president wdt:P27 ?country .
            ?president wdt:P106 ?occupation .
            ?country <http://schema.org/name> ?countryName.
            ?occupation <http://schema.org/name> ?occName.
            ?president <http://schema.org/name> ?presName.
            
            FILTER ( ( !REGEX(?occName, ".*[Pp]resid.*") ) )
            
            }
            GROUP BY ?work ?presName
            ORDER BY DESC (?num_works)
                """;
        Utils u = new Utils();
        u.create_algebra_test(query);
    }
}
