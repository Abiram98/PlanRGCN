package com.org;

import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import com.google.gson.Gson;
import com.google.gson.internal.LinkedTreeMap;

public class DistanceLoader {
    HashMap<String, String> map = new HashMap<>();

    @SuppressWarnings("rawtypes")
    public void loadFile(String file) {
        try {
            Gson gson = new Gson();
            Reader reader = Files.newBufferedReader(Paths.get(file));
            ArrayList<?> list = gson.fromJson(reader, ArrayList.class);
            for (Object entry : list) {
                entry = (LinkedTreeMap) entry;
                System.out.println(entry);
            }
            reader.close();
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }
}
