package com.org;

import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.apache.commons.collections.iterators.IteratorChain;

import com.opencsv.CSVReader;
import com.opencsv.CSVParserBuilder;
import com.opencsv.CSVReaderBuilder;

public class QueryLogReader {
    public String filePath;
    public ArrayList<String> queryIDs;

    public QueryLogReader(String file) {
        filePath = file;
    }

    public void read() throws IOException {
        CSVReader c = new CSVReaderBuilder(new FileReader(filePath))
                .withCSVParser(new CSVParserBuilder()
                        .withQuoteChar('\"')
                        .withSeparator('\t')
                        .build())
                .build();
        ArrayList<String> header = new ArrayList<>(List.of(c.readNext()));

        int idx = -1;
        for (int i = 0; i < header.size(); i++) {
            if (header.get(i).equals("id")) {
                idx = i;
            }
        }
        ArrayList<String> ids = new ArrayList<>();
        Iterator iter = c.iterator();
        while (iter.hasNext()) {
            String[] strings = (String[]) iter.next();
            ids.add(strings[idx]);
        }
        queryIDs = ids;
    }

}
