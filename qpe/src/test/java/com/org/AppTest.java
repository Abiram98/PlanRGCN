package com.org;

import static org.junit.Assert.assertTrue;

import java.io.FileNotFoundException;

import org.apache.jena.query.Query;
import org.apache.jena.query.QueryFactory;
import org.apache.jena.sparql.algebra.Algebra;
import org.apache.jena.sparql.algebra.Op;
import org.junit.Test;

import com.org.Algebra.ExecutionPlanVisitor;

/**
 * Unit test for simple App.
 */
public class AppTest 
{
    /**
     * Rigorous Test :-)
     */
    @Test
    public void shouldAnswerWithTrue()
    {
        assertTrue( true );
    }

    @Test
    public void qpGen(){

        String query = """
            SELECT * WHERE {
                ?s ?p 2 .
            }
        """;
        Query q = QueryFactory.create(query);
        Op o = Algebra.compile(q);
        o = Algebra.optimize(o);
        try {
            ExecutionPlanVisitor visitor = new ExecutionPlanVisitor("/tmp/test.json");
            o.visit(visitor);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }
}
