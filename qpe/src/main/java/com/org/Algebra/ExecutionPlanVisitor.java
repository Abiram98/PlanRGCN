package com.org.Algebra;


import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.Iterator;

import org.apache.jena.graph.Triple;
import org.apache.jena.sparql.algebra.Op;
import org.apache.jena.sparql.algebra.OpVisitorByType;
import org.apache.jena.sparql.algebra.op.Op0;
import org.apache.jena.sparql.algebra.op.Op1;
import org.apache.jena.sparql.algebra.op.Op2;
import org.apache.jena.sparql.algebra.op.OpBGP;
import org.apache.jena.sparql.algebra.op.OpFilter;
import org.apache.jena.sparql.algebra.op.OpJoin;
import org.apache.jena.sparql.algebra.op.OpLeftJoin;
import org.apache.jena.sparql.algebra.op.OpN;
import org.apache.jena.sparql.algebra.op.OpPath;
import org.apache.jena.sparql.algebra.op.OpProcedure;
import org.apache.jena.sparql.algebra.op.OpPropFunc;
import org.apache.jena.sparql.algebra.op.OpSequence;

import org.apache.jena.graph.Node;


/*
 * Output format
 * {
 *  operator name : "name"
 * child : [ {}]
 * }
 * 
 * 
 */

public class ExecutionPlanVisitor extends OpVisitorByType {
    PrintStream stream;
    public int indent = 0;

    private void print(String text){
        String indentStr = "";
        int t = indent;
        while (t > 0){
            indentStr +="  ";
            t--;
        }
        this.stream.println(indentStr+text);
    }
    private void openScope(){
        indent++;
    }
    private void closeScope(){
        indent--;
    }

    public ExecutionPlanVisitor(){
        this.stream = System.out;

    }

    public ExecutionPlanVisitor(String path) throws FileNotFoundException{
        this.stream = new PrintStream(new FileOutputStream(path));
        
    }

    public void visit(Triple triple){
        String t = "{\"Subject\": \""+triple.getSubject();
        t += "\", \"Predicate\": \""+triple.getPredicate();
        t+= "\", \"Object\": \""+triple.getObject();
        t+= "\", \"opName\": \"Triple\"}";
        print(t);
    }

    @Override 
    public void visit(OpBGP opBGP)                   {
        print("{\"opName\":\"BGP\", \"subOp\": [");
        openScope();
        Iterator<Triple> iter= opBGP.getPattern().iterator();
        while(iter.hasNext()){
            visit(iter.next());
            if (iter.hasNext())
                print(",");
        }
        closeScope();
        print("]}");
    }
    
    public void visit(Node n){
        print("Beginning node");
        //print(n.getName());
        print("Ending Node");

    }

    /*@Override 
    public void visit(OpTriple opTriple)              {
        print("Triple");
        print("Subject: "+opTriple.getTriple().getSubject());
        print("Predicate: "+opTriple.getTriple().getPredicate());
        print("Object: "+opTriple.getTriple().getObject());
        print("End Triple");

    }*/


    @Override 
    public void visit(OpPath opPath)                  {
        StringBuilder t = new StringBuilder();
        t.append("{\"opName\": \"")
            .append(opPath.getName())
            .append('"')
            .append(", \"Subject\": \"")
            .append(opPath.getTriplePath().getSubject())
            .append('"')
            .append(", \"Object\": \"")
            .append(opPath.getTriplePath().getObject())
            .append('"')
            .append(", \"Predicate Path\": \"")
            .append(opPath.getTriplePath().getPath().toString())
            .append('"')
            .append('}')
            ;
        print(t.toString());
    }

    @Override 
    public void visit(OpProcedure opProc)             {

    }
    
    @Override 
    public void visit(OpPropFunc opPropFunc)          {

    }

    @Override 
    public void visit(OpJoin opJoin)                  {

    }

    @Override 
    public void visit(OpSequence opSequence)          {
        print("{\"opName\": \""+opSequence.getName()+ "\", \"subOp\": [");
        openScope();
        //List<Op> lst = opSequence.getElements();
        Iterator<Op> iter = opSequence.iterator();
        while(iter.hasNext()){
            Op t = iter.next();
            openScope();
            t.visit(this);
            if (iter.hasNext()){
                print(",");
            }
            closeScope();

        }
        /* 
        for (Op t : lst){
            
            openScope();
            t.visit(this);
            print(",");
            closeScope();
        }*/
        closeScope();
        print("]}");
    }
    
    
    @Override 
    public void visit(OpFilter opFilter)              {
        print("{\"opName\": \""+opFilter.getName()+"\" , \"subOp\": [");
        openScope();
        //TODO: add expressions to queryplan
        opFilter.getSubOp().visit(this);
        closeScope();
        print("]}");
    }

    @Override
    protected void visitN(OpN op) {
        for(Op t : op.getElements()){
            t.visit(this);
        }
        // TODO Auto-generated method stub
        //throw new UnsupportedOperationException("Unimplemented method 'visitN'");
    }
    @Override
    protected void visit2(Op2 op) {
        op.getLeft().visit(this);
        op.getRight().visit(this);
        // TODO Auto-generated method stub
        //throw new UnsupportedOperationException("Unimplemented method 'visit2'");
    }
    @Override
    protected void visit1(Op1 op) {
        op.getSubOp().visit(this);
        // TODO Auto-generated method stub
        //throw new UnsupportedOperationException("Unimplemented method 'visit1'");
    }
    @Override
    protected void visit0(Op0 op) {
        // TODO Auto-generated method stub
        //throw new UnsupportedOperationException("Unimplemented method 'visit0'");
    }
    @Override
    protected void visitFilter(OpFilter op) {
        op.getSubOp().visit(this);
        // TODO Auto-generated method stub
        
        //throw new UnsupportedOperationException("Unimplemented method 'visitFilter'");
    }
    @Override
    protected void visitLeftJoin(OpLeftJoin op) {
        op.getLeft().visit(this);
        op.getRight().visit(this);
        // TODO Auto-generated method stub
        //throw new UnsupportedOperationException("Unimplemented method 'visitLeftJoin'");
    }

    /*@Override 
    public void visit(OpTopN opTop)                   {

    }*/
    
}
