package  com.org;

import java.io.FileNotFoundException;
import java.util.Iterator;
import java.util.LinkedList;

import com.org.Algebra.EntityLogExtractor;
import com.org.Algebra.PredicateLogExtractor;
import com.org.Algebra.Utils;
import com.org.QueryReader.LSQreader;

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
                try {
                    LSQreader reader = new LSQreader(args[1]);
                    
                    Utils u = new Utils();
                    u.extract_query_plan(reader, args[2]);
                } catch (FileNotFoundException e) {
                    // TODO Auto-generated catch block
                    e.printStackTrace();
                }
                break;
            }
            case "extract-predicates-query-log":{
                PredicateLogExtractor.run(args[1], args[2]);
                break;
            }
            case "extract-entity-query-log":{
                EntityLogExtractor.run(args[1], args[2]);
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
                Minus {
                    ?president wdt:P27 ?country .
                }
                ?president wdt:P106 ?occupation .
                Optional {
                    ?country <http://schema.org/name> ?countryName.
                }
                ?occupation <http://schema.org/name> ?occName.
                ?president <http://schema.org/name> ?presName.
                
                FILTER ( ( !REGEX(?occName, ".*[Pp]resid.*") ) )
                
            }
            GROUP BY ?work ?presName
            ORDER BY DESC (?num_works)
            """;
        query="""
            PREFIX dbpr: <http://dbpedia.org/resource/> SELECT * WHERE { { dbpr:Category:1987_IBF_World_Championships ?po ?x } UNION { ?x ?pi dbpr:Category:1987_IBF_World_Championships } } ORDER BY ?pi ?po ?x
                """;
            Utils u = new Utils();
        u.create_algebra_test(query);
    }
}
