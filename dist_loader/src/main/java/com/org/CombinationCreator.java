package com.org;

import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;

import com.google.gson.Gson;
import com.opencsv.CSVParserBuilder;
import com.opencsv.CSVReader;
import com.opencsv.CSVReaderBuilder;

public class CombinationCreator {
    public String filePath;
    public String combinationFile;
    public String combDir;
    public ArrayList<String> queryIDs;
    public HashMap<String, String> id2query;
    int limit = 2000;

    public CombinationCreator(String file, String combFile, String combDirr) {
        filePath = file;
        combinationFile = combFile;
        combDir = combDirr;
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
        int strIdx = -1;
        for (int i = 0; i < header.size(); i++) {
            if (header.get(i).equals("id")) {
                idx = i;
            }
            if (header.get(i).equals("queryString")) {
                strIdx = i;
            }
        }
        id2query = new HashMap<>();
        ArrayList<String> ids = new ArrayList<>();
        Iterator iter = c.iterator();
        while (iter.hasNext()) {
            String[] strings = (String[]) iter.next();
            id2query.put(strings[idx], strings[strIdx]);
        }
        queryIDs = ids;
    }

    public HashMap<String, Object> getJsonMap(String queryID1, String queryString1, String queryID2,
            String queryString2) {
        Gson gson = new Gson();
        HashMap<String, Object> map = new HashMap<>();
        map.put("queryID1", queryID1);
        map.put("queryID2", queryID2);
        map.put("queryString1", queryString1);
        map.put("queryString2", queryString2);
        return map;
        // return gson.toJson(map);
    }

    public static void run(String file, String combFile, String combDir) throws IOException {
        // file : /data/DBpedia2016_0_1_10_weight_loss/train_sampled.tsv
        // combFile: /data/dbpedia_dist2bak/combinations/comb1.csv
        CombinationCreator creator = new CombinationCreator(file, combFile, combDir);
        creator.read();
        creator.process();
    }

    public void process() throws IOException {
        CSVReader c = new CSVReaderBuilder(new FileReader(combinationFile))
                .withCSVParser(new CSVParserBuilder()
                        .withQuoteChar('\"')
                        .withSeparator(',')
                        .build())
                .build();

        ArrayList<HashMap<String, Object>> list = new ArrayList<>();
        Iterator iter = c.iterator();
        int f_count = 0;
        while (iter.hasNext()) {
            String[] strings = (String[]) iter.next();
            String queryID1 = strings[0];
            String queryID2 = strings[1];
            String queryString1 = id2query.get(queryID1);
            String queryString2 = id2query.get(queryID2);
            list.add(getJsonMap(queryID1, queryString1, queryID2, queryString2));
            if (list.size() % limit == 0) {
                Gson gson = new Gson();
                FileWriter fw = new FileWriter(combDir + "/comb" + f_count + ".json");
                gson.toJson(list, fw);
                fw.flush();
                fw.close();
                f_count++;
                list = new ArrayList<>();
            }
        }
    }
}
