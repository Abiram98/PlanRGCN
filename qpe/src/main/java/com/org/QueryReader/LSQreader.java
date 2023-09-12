package com.org.QueryReader;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

import com.opencsv.CSVParserBuilder;
import com.opencsv.CSVReader;
import com.opencsv.CSVReaderBuilder;

public class LSQreader {
    String path;
    LinkedList<String> queries = new LinkedList<>();
    LinkedList<String> ids = new LinkedList<>();

    public LinkedList<String> getQueries() {
        return queries;
    }
    public LinkedList<String> getIds() {
        return ids;
    }

    public LSQreader(String file_path) throws FileNotFoundException{
        path = file_path;
        //CSVReader reader = new CSVReader(new FileReader(App.class.getClassLoader().getResource("csv.csv").getFile()), ',','"','-');
        CSVReader c = new CSVReaderBuilder(new FileReader(file_path))
                .withCSVParser(new CSVParserBuilder()
                        .withQuoteChar('\"')
                        .withSeparator('\t')
                        .build())
                .build();
        //CSVParser parser = new CSVParser(c);
        try{
            List<String[]> allRows = c.readAll();
            Iterator<String[]> iter = allRows.iterator();
            iter.next();
            while( iter.hasNext()){
                String[] i = iter.next();
                queries.add(i[1]);
                ids.add(i[0]);
            }
        
        }catch (Exception e){
            e.printStackTrace();
        }
    }
}
