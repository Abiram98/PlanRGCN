package com.org.Algebra;

import org.apache.jena.sparql.algebra.*;
import org.apache.jena.sparql.sse.SSE;

import com.org.QueryReader.LSQreader;

import java.io.FileNotFoundException;
import java.util.concurrent.TimeUnit;

import org.apache.commons.lang3.time.StopWatch;
import org.apache.jena.query.Query;
import org.apache.jena.query.QueryFactory;

public class Utils {
    public static boolean sub_id = false; 

    public void create_algebra_test(String query) {
        Query q = QueryFactory.create(query);
        Op o = Algebra.compile(q);
        o = Algebra.optimize(o);
        SSE.write(o);
        try {
            ExecutionPlanVisitor visitor = new ExecutionPlanVisitor("/PlanRGCN/test.json");
            // o.visit(visitor);
            // o.visit(new CustomWalker(visitor));
            o.visit(visitor);
            // OpWalker.walk(o, visitor);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    public void extract_query_plan(String query, String filepath) {
        Query q = QueryFactory.create(query);
        Op o = Algebra.compile(q);
        o = Algebra.optimize(o);
        //SSE.write(o);
        try {
            ExecutionPlanVisitor visitor = new ExecutionPlanVisitor(filepath);
            o.visit(visitor);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    public long time_query_plan(String query) {
        StopWatch watch = new StopWatch();
        watch.start();
        Query q = QueryFactory.create(query);
        Op o = Algebra.compile(q);
        o = Algebra.optimize(o);
        // SSE.write(o);
        ExecutionPlanVisitor visitor = new ExecutionPlanVisitor(false);
        o.visit(visitor);
        watch.stop();
        long time = watch.getTime(TimeUnit.MILLISECONDS);
        return time;
    }

    /*
     * The extracted queries are written to files with
     */
    public void extract_query_plan(LSQreader reader, String outputdir) {
        java.util.ArrayList<String> data = reader.getNext();
        while(data != null){
            String id = data.get(0);
            String query = data.get(1);
            try{
                System.out.println("Extracting for "+id);
            if(sub_id){
                	extract_query_plan(query, outputdir + "/" + id.substring(20));
            }else{
                        extract_query_plan(query, outputdir + "/" + String.valueOf(id));
            }}catch (org.apache.jena.query.QueryException e) {
                System.out.println("Did not work for " + id);
            }
            data = reader.getNext();
        }
        /*
        LinkedList<String> ids = reader.getIds();
        LinkedList<String> queries = reader.getQueries();
        for (int i = 0; i < ids.size(); i++) {
            try {
		if(sub_id){
                	extract_query_plan(queries.get(i), outputdir + "/" + ids.get(i).substring(20));
		}else{
                	extract_query_plan(queries.get(i), outputdir + "/" + String.valueOf(ids.get(i)));
		}
		} catch (org.apache.jena.query.QueryException e) {
                System.out.println("Did not work for " + ids.get(i));
            }
        } */
    }

}
