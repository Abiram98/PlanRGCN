package com.org.Algebra;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintStream;
import org.apache.jena.sparql.algebra.Op;
import org.apache.jena.sparql.algebra.OpVisitorBase;
import org.apache.jena.sparql.algebra.OpWalker;

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
