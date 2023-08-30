package com.org.Algebra;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.Iterator;

import org.apache.jena.graph.Triple;
import org.apache.jena.sparql.algebra.Op;
import org.apache.jena.sparql.algebra.OpVisitorBase;
import org.apache.jena.sparql.algebra.OpVisitorByType;
import org.apache.jena.sparql.algebra.OpWalker;
import org.apache.jena.sparql.algebra.op.Op0;
import org.apache.jena.sparql.algebra.op.Op1;
import org.apache.jena.sparql.algebra.op.Op2;
import org.apache.jena.sparql.algebra.op.OpFilter;
import org.apache.jena.sparql.algebra.op.OpLeftJoin;
import org.apache.jena.sparql.algebra.op.OpN;

public class ExecutionPlanVisitor2 extends OpVisitorBase{
    PrintStream stream;
    public int indent = 0;

    public ExecutionPlanVisitor2(){
        this.stream = System.out;

    }

    public ExecutionPlanVisitor2(String path) throws FileNotFoundException{
        this.stream = new PrintStream(new FileOutputStream(path));
        
    }

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
    void opVisitorWalker(Op op) {
    OpWalker.walk(op, this);
    }
    

}
