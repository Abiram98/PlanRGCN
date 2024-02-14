package com.org;

import java.io.FileWriter;
import java.io.IOException;
import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import org.apache.commons.lang3.time.StopWatch;

import com.google.gson.Gson;
import com.google.gson.internal.LinkedTreeMap;

public class DistanceLoader {
    HashMap<StringPair, double[]> map = new HashMap<>();

    @SuppressWarnings("rawtypes")
    public void loadFile(String file) {
        StopWatch watch = new StopWatch();
        watch.start();
        try {
            Gson gson = new Gson();
            Reader reader = Files.newBufferedReader(Paths.get(file));
            ArrayList<?> list = gson.fromJson(reader, ArrayList.class);
            for (Object entry : list) {
                String queryID2 = (String) ((LinkedTreeMap) entry).get("queryID2");
                String queryID1 = (String) ((LinkedTreeMap) entry).get("queryID1");
                double dist = Double.parseDouble((String) ((LinkedTreeMap) entry).get("dist"));
                double time = Double.parseDouble((String) ((LinkedTreeMap) entry).get("time")) / 1000.0;
                this.map.put(new StringPair(queryID1, queryID2), new double[] { dist, time });
            }
            reader.close();
        } catch (Exception ex) {
        }
    }

    public double get(String queryID1, String queryID2) {
        double[] vals = this.map.get(new StringPair(queryID1, queryID2));
        if (vals == null) {
            return -1;
        } else {
            return vals[1];
        }
    }

    public double get(String queryID1, String queryID2, FileWriter writer) throws IOException {
        double val = get(queryID1, queryID2);
        if (val == -1) {
            writer.write(queryID1 + "," + queryID2 + "\n");
        }
        return val;
    }
}
