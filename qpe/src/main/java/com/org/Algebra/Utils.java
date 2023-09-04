package com.org.Algebra;
import org.apache.jena.sparql.algebra.*;
import org.apache.jena.sparql.sse.SSE;

import java.io.FileNotFoundException;

import org.apache.jena.query.Query;
import org.apache.jena.query.QueryFactory;

public class Utils{

    public void create_algebra_test(String query){
        Query q = QueryFactory.create(query);
        Op o = Algebra.compile(q);
        o = Algebra.optimize(o);
        SSE.write(o);
        try {
            ExecutionPlanVisitor visitor = new ExecutionPlanVisitor("/PlanRGCN/test.json");
            //o.visit(visitor);
            //o.visit(new CustomWalker(visitor));
            o.visit(visitor);
            //OpWalker.walk(o, visitor);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }
    public void extract_query_plan(){

    }
    
}