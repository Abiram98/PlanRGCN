package com.org;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.TimeUnit;

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

    public void loadFile(String file, Set<String> ids) {
        StopWatch watch = new StopWatch();
        watch.start();
        try {
            Gson gson = new Gson();
            Reader reader = Files.newBufferedReader(Paths.get(file));
            ArrayList<?> list = gson.fromJson(reader, ArrayList.class);
            for (Object entry : list) {
                String queryID2 = (String) ((LinkedTreeMap) entry).get("queryID2");
                String queryID1 = (String) ((LinkedTreeMap) entry).get("queryID1");
                if ((!ids.contains(queryID1)) || (!ids.contains(queryID2))) {
                    continue;
                }
                double dist = Double.parseDouble((String) ((LinkedTreeMap) entry).get("dist"));
                double time = Double.parseDouble((String) ((LinkedTreeMap) entry).get("time")) / 1000.0;
                this.map.put(new StringPair(queryID1, queryID2), new double[] { dist, time });
            }
            reader.close();
        } catch (Exception ex) {
        }
    }

    public double get(String queryID1, String queryID2) {
        if (queryID1.equals(queryID2)) {
            return 0;
        }
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

    public static void runComputationCalculator(String queryLog, String distanceDist, String missPairs) {
        QueryLogReader reader = new QueryLogReader(queryLog);
        try {
            reader.read();
        } catch (IOException e) {
            e.printStackTrace();
        }
        Set<String> idSet = new HashSet<>(reader.queryIDs);
        DistanceLoader distLoader = new DistanceLoader();
        File f = new File(distanceDist);
        try {
            FileWriter fw = new FileWriter(missPairs);
            int i = 0;
            StopWatch watch = new StopWatch();
            watch.start();
            for (String s : f.list()) {
                distLoader.loadFile(f.getPath() + "/" + s, idSet);
                i++;
                System.out.println("File " + i + ": " + ((double) watch.getTime(TimeUnit.MILLISECONDS)) / 1000 + "\n");
            }
            ArrayList<String> queryIds = reader.queryIDs;
            double total = 0;
            int missingPairs = 0;
            for (int j = 0; j < queryIds.size(); j++) {
                for (int v = j + 1; v < queryIds.size(); v++) {
                    double val = distLoader.get(queryIds.get(j), queryIds.get(v), fw);
                    if (val == -1) {
                        missingPairs++;
                    } else {
                        total += val;
                    }
                }
            }
            fw.close();
            System.out.println("Total amount time spent on dist calulcations on training");
            System.out.println(total);
            System.out.println("Missing Pairs");
            System.out.println(missingPairs);
            System.out.println("Distance HashSet size: " + distLoader.map.size());
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }
}
