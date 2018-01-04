import java.io.*;
import java.util.*;

class Attribute {
    int article_id;
    int dimension;
    double value;

    Attribute(int article_id, int dimension, double value) {
        this.article_id = article_id;
        this.dimension = dimension;
        this.value = value;
    }

    Attribute(int dimension, double value) {
        this.dimension = dimension;
        this.value = value;
    }
}

class Result {
    double criterion_value;
    double entropy;
    double purity;
    HashMap<Integer,ArrayList<Integer>> clusters;
    int trail;
    HashMap<Integer,HashMap<String,Integer>> class_distribution;
}

class clustering {
    HashMap<Integer,ArrayList<Attribute>> data_set;
    ArrayList<Integer> overall_articles;
    HashMap<Integer,String> id_label_map;
    int num_clusters;
    String criterion_function;
    int trails;
    HashMap<Integer,ArrayList<Attribute>> centroids = new HashMap<>();
    Random rand;
    int[] seeds = {1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39};
    HashMap<Integer,ArrayList<Integer>> cluster_article_map = new HashMap<>();
    HashMap<Integer,Integer> article_cluster_map = new HashMap<>();
    HashMap<Integer,ArrayList<Attribute>> cluster_composite_map = new HashMap<>();
    HashMap<Integer,Double> cluster_composite_norm_map = new HashMap<>();
    ArrayList<Attribute> overall_composite_vector;
    int max_id;
    ArrayList<Attribute> overall_centroid;


    clustering(HashMap<Integer,ArrayList<Attribute>> data_set, ArrayList<Integer> overall_articles, HashMap<Integer,String> id_label_map, int num_clusters, String criterion_function, int trails, int max_id) {
        this.data_set = data_set;
        this.overall_articles = overall_articles;
        this.id_label_map = id_label_map;
        this.num_clusters = num_clusters;
        this.criterion_function = criterion_function;
        this.trails = trails;
        this.max_id = max_id;
    }

    private void initializeCentroids(int trail) {
        centroids.clear();
        cluster_article_map.clear();
        int seed = seeds[trail % 20];
        rand = new Random(seed);
        for(int cluster_index = 0; cluster_index < num_clusters; cluster_index++) {
            int random_article = rand.nextInt(this.max_id);
            while(true) {
                if(data_set.containsKey(random_article) && !centroids.containsKey(random_article)) {
                    break;
                }
                random_article = rand.nextInt(this.max_id);
            }
            //System.out.println("Cluster :  "+cluster_index+" Centroid article : "+random_article);
            centroids.put(cluster_index,data_set.get(random_article));
            cluster_article_map.put(cluster_index,new ArrayList<Integer>());
        }
    }

    private int getClosestCentroid(ArrayList<Attribute> attributes) {
        double closest_value = Double.MAX_VALUE;
        if(criterion_function.equalsIgnoreCase("I2") || criterion_function.equalsIgnoreCase("E1")) {
            closest_value = Double.MIN_VALUE;
        }
        int closest_centroid = 0;
        for(int current_centroid : centroids.keySet()) {
            double value = getCloseness(centroids.get(current_centroid), attributes, criterion_function);
            if(criterion_function.equalsIgnoreCase("SSE") && value < closest_value) {
                closest_value = value;
                closest_centroid = current_centroid;
            }
            if((criterion_function.equalsIgnoreCase("I2") || criterion_function.equalsIgnoreCase("E1")) && value > closest_value) {
                closest_value = value;
                closest_centroid = current_centroid;
            }
        }
        return closest_centroid;
    }

    private double getCloseness(ArrayList<Attribute> vector1, ArrayList<Attribute> vector2, String criterion_function) {
        double result = 0;

        if(criterion_function.equalsIgnoreCase("SSE")) {
            int index1 = 0, index2 = 0;
            int size1 = vector1.size(), size2 = vector2.size();
            double sum = 0;
            while(index1 < size1 && index2 < size2) {
                if(vector1.get(index1).dimension < vector2.get(index2).dimension) {
                    sum += Math.pow(vector1.get(index1).value, 2);
                    index1++;
                } else if(vector1.get(index1).dimension > vector2.get(index2).dimension) {
                    sum += Math.pow(vector2.get(index2).value, 2);
                    index2++;
                } else {
                    sum += Math.pow((vector1.get(index1).value - vector2.get(index2).value), 2);
                    index1++;
                    index2++;
                }
            }
            while(index1 < size1) {
                sum += Math.pow(vector1.get(index1).value, 2);
                index1++;
            }
            while(index2 < size2) {
                sum += Math.pow(vector2.get(index2).value, 2);
                index2++;
            }
            result = sum;
        } else {
            int index1 = 0, index2 = 0;
            int size1 = vector1.size(), size2 = vector2.size();
            double product = 0;
            double sum1 = 0;
            double sum2 = 0;
            while(index1 < size1 && index2 < size2) {
                if(vector1.get(index1).dimension < vector2.get(index2).dimension) {
                    sum1 += Math.pow(vector1.get(index1).value, 2);
                    index1++;
                } else if(vector1.get(index1).dimension > vector2.get(index2).dimension) {
                    sum2 += Math.pow(vector2.get(index2).value, 2);
                    index2++;
                } else {
                    product += (vector1.get(index1).value * vector2.get(index2).value);
                    sum1 += Math.pow(vector1.get(index1).value, 2);
                    sum2 += Math.pow(vector2.get(index2).value, 2);
                    index1++;
                    index2++;
                }
            }
            while(index1 < size1) {
                sum1 += Math.pow(vector1.get(index1).value, 2);
                index1++;
            }
            while(index2 < size2) {
                sum2 += Math.pow(vector2.get(index2).value, 2);
                index2++;
            }

            result = (product / (Math.sqrt(sum1) * Math.sqrt(sum2)));
        }
        return result;
    }

    private void updateCentroids() {
        centroids.clear();

        for(int cluster : cluster_article_map.keySet()) {

            // Find the count and sum of all the values of each dimension of all the articles present in this cluster
            TreeMap<Integer, Double> dimensions_sum = new TreeMap<>();

            for(Integer article : cluster_article_map.get(cluster)) {
                for(Attribute attr : data_set.get(article)) {
                    if(!dimensions_sum.containsKey(attr.dimension)) {
                        dimensions_sum.put(attr.dimension, 0.0);
                    }
                    dimensions_sum.put(attr.dimension,dimensions_sum.get(attr.dimension) + attr.value);
                }
            }

            // Find the average value of each dimension in this cluster which is the new centroid
            ArrayList<Attribute> new_centroid = new ArrayList<>();

            for(Integer dimension : dimensions_sum.keySet()) {
                double average = dimensions_sum.get(dimension) / (double)cluster_article_map.get(cluster).size();
                new_centroid.add(new Attribute(dimension,average));
            }

            centroids.put(cluster,new_centroid);
        }
    }

    private boolean is_converged(HashMap<Integer,ArrayList<Attribute>>  old_centroids, double threshold) {

        double max_distance = Double.MIN_VALUE;

        for(Integer current_centroid : old_centroids.keySet()) {
            double value = getCloseness(old_centroids.get(current_centroid), centroids.get(current_centroid), "SSE");
            if(max_distance < value) {
                max_distance = value;
            }
        }
        //System.out.println("max_distance b/w centroids : "+max_distance);
        if(max_distance < threshold) {
            return true;
        }

        return false;
    }

    private double getValueOfObjectiveFunction() {
        double result = 0.0;

        if(criterion_function.equalsIgnoreCase("SSE") || criterion_function.equalsIgnoreCase("I2")) {
            for(int cluster : cluster_article_map.keySet()) {
                for(Integer article : cluster_article_map.get(cluster)) {
                    result += getCloseness(centroids.get(cluster), data_set.get(article), criterion_function);
                }
            }
        } else {
            for(int cluster : cluster_article_map.keySet()) {
                result += ((double)cluster_article_map.get(cluster).size() * getCloseness(centroids.get(cluster), overall_centroid, criterion_function));
            }
        }
        return result;
    }

    private void calculateEntropyAndPurity(Result result) throws IOException {

        HashMap<Integer,HashMap<String,Integer>> class_distribution = new HashMap<>();
        int overall_total_articles = 0;
        double overall_entropy = 0;
        double overall_purity = 0;

        for(int cluster : cluster_article_map.keySet()) {

            HashMap<String,Integer> class_count = new HashMap<>();

            for(Integer article : cluster_article_map.get(cluster)) {
                String label = id_label_map.get(article);
                if(!class_count.containsKey(label)) {
                    class_count.put(label,0);
                }
                class_count.put(label,class_count.get(label) + 1);
            }
            class_distribution.put(cluster,class_count);
            overall_total_articles += cluster_article_map.get(cluster).size();
        }

        for(int cluster : cluster_article_map.keySet()) {

            int total_articles = cluster_article_map.get(cluster).size();
            HashMap<String,Integer> class_count = class_distribution.get(cluster);
            double cluster_entropy = 0;
            double cluster_purity = Double.MIN_VALUE;

            for(String class_label: class_count.keySet()) {
                double value = ((double)class_count.get(class_label) / (double)total_articles);
                cluster_entropy += ((value * (Math.log(value) / Math.log(2))));
                if(cluster_purity < value) {
                    cluster_purity = value;
                }
            }

            cluster_entropy = -1.0 * cluster_entropy;

            overall_entropy += (((double)total_articles / (double)overall_total_articles) * cluster_entropy);
            overall_purity += (((double)total_articles / (double)overall_total_articles) * cluster_purity);
        }

        System.out.println("overall_entropy : "+overall_entropy);
        System.out.println("overall_purity : "+overall_purity);


        //code to write the class distribution to a file
        ArrayList<String> topics = new ArrayList<>(Arrays.asList("earn", "acq", "crude", "trade", "money-fx", "interest",
                "ship", "sugar", "coffee", "gold", "money-supply", "gnp", "cpi", "cocoa", "copper", "jobs", "iron-steel",
                "alum", "grain", "reserves"));
        BufferedWriter class_dist_writer = new BufferedWriter(new FileWriter("class_distribution.csv"));
        class_dist_writer.write("Cluster,");
        for(String topic : topics) {
            class_dist_writer.write(topic+",");
        }
        class_dist_writer.newLine();
        for(Integer cluster : class_distribution.keySet()) {
            class_dist_writer.write(cluster+",");
            for(String class_label: topics) {
                Integer value = class_distribution.get(cluster).get(class_label);
                if(value == null) {
                    value = 0;
                }
                class_dist_writer.write(value+",");
            }
            class_dist_writer.newLine();
        }
        class_dist_writer.close();


        result.entropy = overall_entropy;
        result.purity = overall_purity;
        result.class_distribution = class_distribution;
    }

    public ArrayList<Attribute> calculateOverallCentroid() {
        double total_size = overall_articles.size();
        // Find the count and sum of all the values in the dataset
        TreeMap<Integer, Double> dimensions_sum = new TreeMap<>();

        for(Integer article : overall_articles) {
            for (Attribute attr : data_set.get(article)) {
                if (!dimensions_sum.containsKey(attr.dimension)) {
                    dimensions_sum.put(attr.dimension, 0.0);
                }
                dimensions_sum.put(attr.dimension, dimensions_sum.get(attr.dimension) + attr.value);
            }
        }

        ArrayList<Attribute> result = new ArrayList<>();

        for(Integer dimension : dimensions_sum.keySet()) {
            result.add(new Attribute(dimension,(dimensions_sum.get(dimension) / total_size)));
        }

        return result;
    }


    public Result alternateLeastSquareClustering() throws IOException {

        Result result = new Result();

        double best_solution = Double.MAX_VALUE;
        if(criterion_function.equalsIgnoreCase("I2")) {
            best_solution = Double.MIN_VALUE;
        }
        HashMap<Integer,ArrayList<Integer>> best_clustering = new HashMap<>();
        int best_trail = 0;

        if(criterion_function.equalsIgnoreCase("E1")) {
            overall_centroid = calculateOverallCentroid();
        }

        for(int trail = 0; trail < trails; trail++) {

            initializeCentroids(trail);

            int iter = 0;

            while (true) {

                for (Integer article : cluster_article_map.keySet()) {
                    cluster_article_map.get(article).clear();
                }

                // assign articles to the closest centroids
                for (Integer article : overall_articles) {
                    int closest_centroid = getClosestCentroid(data_set.get(article));
                    cluster_article_map.get(closest_centroid).add(article);
                }

                // Make a copy of old centroids before updating centroids
                HashMap<Integer, ArrayList<Attribute>> old_centroids = new HashMap(centroids);

                // update centroids
                updateCentroids();

                //check for convergence
                double threshold = 0.001;
                if (is_converged(old_centroids, threshold)) {
                    break;
                }

                iter++;
                System.out.println("Trail : "+trail+" iteration : "+iter+" completed");
                if (iter == 10) {
                    break;
                }
            }

            double value = getValueOfObjectiveFunction();
            if ((criterion_function.equalsIgnoreCase("SSE") || criterion_function.equalsIgnoreCase("E1")) && value < best_solution) {
                best_solution = value;
                best_clustering = new HashMap(cluster_article_map);
                best_trail = trail;
            } else if (criterion_function.equalsIgnoreCase("I2") && value > best_solution) {
                best_solution = value;
                best_clustering = new HashMap(cluster_article_map);
                best_trail = trail;
            }

            System.out.println("Trail : "+trail+" completed");
        }

        calculateEntropyAndPurity(result);

        result.criterion_value = best_solution;
        result.clusters = best_clustering;
        result.trail = best_trail;

        return result;

    }

    /*FUNCTIONS TO PERFORM INCREMENTAL CLUSTERING*/

    public ArrayList<Attribute> calculateCompositeVector(int cluster) {

        // Find the count and sum of all the values of each dimension of all the articles present in this cluster
        TreeMap<Integer, Double> dimensions_sum = new TreeMap<>();

        for(Integer article : cluster_article_map.get(cluster)) {
            for(Attribute attr : data_set.get(article)) {
                if(!dimensions_sum.containsKey(attr.dimension)) {
                    dimensions_sum.put(attr.dimension, 0.0);
                }
                dimensions_sum.put(attr.dimension,dimensions_sum.get(attr.dimension) + attr.value);
            }
        }

        // Find the average value of each dimension in this cluster which is the new centroid
        ArrayList<Attribute> result = new ArrayList<>();

        for(Integer dimension : dimensions_sum.keySet()) {
            result.add(new Attribute(dimension,dimensions_sum.get(dimension)));
        }

        return result;
    };

    private void editCompositeVector(ArrayList<Attribute> vector, ArrayList<Attribute> article, String edit_type) {

        int index1 = 0, index2 = 0;
        int size1 = vector.size(), size2 = article.size();

        while(index1 < size1 && index2 < size2) {
            if(vector.get(index1).dimension < article.get(index2).dimension) {
                index1++;
            } else if(vector.get(index1).dimension > article.get(index2).dimension) {
                vector.add(index1, new Attribute(article.get(index2).dimension,article.get(index2).value));
                index1++;
                index2++;
            } else {
                if(edit_type.equals("add")) {
                    vector.get(index1).value = vector.get(index1).value + article.get(index2).value;
                } else if(edit_type.equals("remove")) {
                    vector.get(index1).value = vector.get(index1).value - article.get(index2).value;
                }

                index1++;
                index2++;
            }
        }

        if(edit_type.equals("add")) {
            while(index2 < size2) {
                vector.add(index1, new Attribute(article.get(index2).dimension,article.get(index2).value));
                index1++;
                index2++;
            }
        }

    }

    public double calculateNorm(ArrayList<Attribute> vector, int cluster_size) {
        double result = 0;
        if(criterion_function.equalsIgnoreCase("I2")) {
            for(Attribute attr : vector) {
                result += Math.pow(attr.value, 2);
            }
            result = Math.sqrt(result);
        } else if(criterion_function.equalsIgnoreCase("E1")) {
            int size1 = vector.size();
            int size2 = overall_composite_vector.size();
            int index1 = 0, index2 = 0;
            double sum = 0.0;
            double product = 0.0;
            while(index1 < size1 && index2 < size2) {
                if(vector.get(index1).dimension < overall_composite_vector.get(index2).dimension) {
                    sum += Math.pow(vector.get(index1).value, 2);
                    index1++;
                } else if(vector.get(index1).dimension > overall_composite_vector.get(index2).dimension) {
                    index2++;
                } else {
                    product += (vector.get(index1).value * overall_composite_vector.get(index2).value);
                    sum += Math.pow(vector.get(index1).value, 2);
                    index1++;
                    index2++;
                }
            }
            result = (double)cluster_size * (product / Math.sqrt(sum));
        }
        return result;
    }

    public double calculateCriterion() {
        double result = 0;
        for(Integer cluster : cluster_composite_map.keySet()) {
            double norm = calculateNorm(cluster_composite_map.get(cluster), cluster_article_map.get(cluster).size());
            cluster_composite_norm_map.put(cluster,norm);
            result += norm;
        }
        return result;
    }

    public double findReplacementCentroid(Integer article, double criterion_value) {

        double optimised_criterion_value = Double.MAX_VALUE;
        if(criterion_function.equalsIgnoreCase("I2")) {
            optimised_criterion_value = Double.MIN_VALUE;
        }
        ArrayList<Attribute> best_added_composite = new ArrayList<>();
        Integer best_added_cluster = 0;

        int current_cluster = article_cluster_map.get(article);
        ArrayList<Attribute> removed_cluster_composite =  new ArrayList<>(cluster_composite_map.get(current_cluster));
        editCompositeVector(removed_cluster_composite,data_set.get(article),"remove");

        double removed_cluster_composite_norm = calculateNorm(removed_cluster_composite, cluster_article_map.get(current_cluster).size() - 1);

        for(Integer cluster : cluster_composite_map.keySet()) {
            if(cluster != current_cluster) {
                ArrayList<Attribute> added_cluster_composite =  new ArrayList<>(cluster_composite_map.get(cluster));
                editCompositeVector(added_cluster_composite,data_set.get(article),"add");

                double added_cluster_composite_norm = calculateNorm(added_cluster_composite, cluster_article_map.get(current_cluster).size() + 1);

                // subtracting the older values of the two clusters and adding the new values of two clusters
                double updated_criterion_value  = criterion_value - (cluster_composite_norm_map.get(current_cluster) + cluster_composite_norm_map.get(cluster));
                updated_criterion_value += removed_cluster_composite_norm + added_cluster_composite_norm;

                if((criterion_function.equalsIgnoreCase("I2") && optimised_criterion_value < updated_criterion_value) ||
                        (criterion_function.equalsIgnoreCase("E1") && optimised_criterion_value > updated_criterion_value)) {
                    best_added_cluster = cluster;
                    optimised_criterion_value = updated_criterion_value;
                    best_added_composite = added_cluster_composite;
                }
            }
        }

        if((criterion_function.equalsIgnoreCase("I2") && optimised_criterion_value > criterion_value) || (criterion_function.equalsIgnoreCase("E1") && optimised_criterion_value < criterion_value)) {
            cluster_composite_map.put(current_cluster,removed_cluster_composite);
            cluster_composite_map.put(best_added_cluster,best_added_composite);
            article_cluster_map.put(article,best_added_cluster);
            cluster_article_map.get(current_cluster).remove(article);
            cluster_article_map.get(best_added_cluster).add(article);
            return optimised_criterion_value;
        } else {
            return -1.0;
        }
    }

    public ArrayList<Attribute> calculateOverallComposite() {

        // Find the count and sum of all the values in the dataset
        TreeMap<Integer, Double> dimensions_sum = new TreeMap<>();

        for(Integer article : overall_articles) {
            for (Attribute attr : data_set.get(article)) {
                if (!dimensions_sum.containsKey(attr.dimension)) {
                    dimensions_sum.put(attr.dimension, 0.0);
                }
                dimensions_sum.put(attr.dimension, dimensions_sum.get(attr.dimension) + attr.value);
            }
        }

        ArrayList<Attribute> result = new ArrayList<>();

        for(Integer dimension : dimensions_sum.keySet()) {
            result.add(new Attribute(dimension,dimensions_sum.get(dimension)));
        }

        return result;
    }

    public Result incrementalClustering() throws IOException {

        Result result = new Result();

        double best_solution = Double.MAX_VALUE;

        HashMap<Integer,ArrayList<Integer>> best_clustering = new HashMap<>();

        overall_composite_vector = calculateOverallComposite();

        int best_trail = 0;

        for(int trail = 0; trail < trails; trail++) {

            initializeCentroids(trail);

            // assign articles to the closest centroids
            for (Integer article : overall_articles) {
                int closest_centroid = getClosestCentroid(data_set.get(article));
                cluster_article_map.get(closest_centroid).add(article);
                article_cluster_map.put(article,closest_centroid);
            }

            for(Integer cluster : cluster_article_map.keySet()) {
                cluster_composite_map.put(cluster,calculateCompositeVector(cluster));
            }

            double criterion_value = calculateCriterion();

            int iter = 0;

            while (true) {

                int num_of_articles_changed = 0;

                //find a replacement cluster which optimizes the objective function if any
                for (Integer article : overall_articles) {
                    double value = findReplacementCentroid(article, criterion_value);
                    if(value != -1.0) {
                        num_of_articles_changed++;
                        criterion_value = value;
                    }
                }

                iter++;
                System.out.println("Trail : "+trail+" iteration : "+iter+" completed");
                if (iter == 5) {
                    break;
                }
            }


            if (criterion_function.equalsIgnoreCase("E1") && criterion_value < best_solution) {
                best_solution = criterion_value;
                best_clustering = new HashMap(cluster_article_map);
                best_trail = trail;
            } else if (criterion_function.equalsIgnoreCase("I2") && criterion_value > best_solution) {
                best_solution = criterion_value;
                best_clustering = new HashMap(cluster_article_map);
                best_trail = trail;
            }

            System.out.println("Trail : "+trail+" completed");
        }

        calculateEntropyAndPurity(result);

        result.criterion_value = best_solution;
        result.clusters = best_clustering;
        result.trail = best_trail;

        return result;

    }
}

public class kcluster {

    public static void main(String[] args) throws IOException {

        String input_file_name = "freq.csv";
        String class_file_name = "reuters21578.clas";
        String output_file_name = "out.txt";
        int num_clusters = 20;
        String criterion_function = "SSE";
        int trails = 2;

        if(args.length == 6) {
            input_file_name = args[0];
            criterion_function = args[1];
            class_file_name = args[2];
            num_clusters = Integer.parseInt(args[3]);
            trails = Integer.parseInt(args[4]);
            output_file_name = args[5];
        } else {
            System.out.println("Wrong number of arguments. Running basic cases");
        }


        HashMap<Integer,ArrayList<Attribute>> data_set = new HashMap<>();
        ArrayList<Integer> overall_articles = new ArrayList<>();

        String line = null;

        int max_id = Integer.MIN_VALUE;

        HashMap<Integer,String> id_label_map = new HashMap<>();

        BufferedReader bufferedReader = new BufferedReader(new FileReader(class_file_name));

        while((line = bufferedReader.readLine()) != null) {
            String[] split_line = line.split(",");
            int id = Integer.parseInt(split_line[0]);
            String label = split_line[1];
            id_label_map.put(id, label);
            //System.out.println("id : "+id+" label : "+label);
        }

        bufferedReader.close();

        BufferedWriter writer = new BufferedWriter(new FileWriter(output_file_name));

        bufferedReader = new BufferedReader(new FileReader(input_file_name));

        while((line = bufferedReader.readLine()) != null) {
            String[] split_line = line.split(",");
            int id = Integer.parseInt(split_line[0]);
            int dimension = Integer.parseInt(split_line[1]);
            double value = Double.parseDouble(split_line[2]);
            if(!data_set.containsKey(id)) {
                data_set.put(id, new ArrayList<Attribute>());
                overall_articles.add(id);
            }
            max_id = Math.max(max_id,id);
            data_set.get(id).add(new Attribute(dimension,value));
            //System.out.println("id : "+id+" dimension : "+dimension+" value : "+value);
        }

        bufferedReader.close();
        System.out.println("Completed reading data set");

        long start_time = System.currentTimeMillis();

        clustering kcluster = new clustering(data_set,overall_articles, id_label_map, num_clusters, criterion_function, trails, max_id);
        Result result = new Result();

        if(criterion_function.equalsIgnoreCase("SSE") || criterion_function.equalsIgnoreCase("I2")) {
            result = kcluster.alternateLeastSquareClustering();
        } else {
            result = kcluster.incrementalClustering();
        }

        long end_time = System.currentTimeMillis();

        for(Integer cluster : result.clusters.keySet()) {
            for(Integer article : result.clusters.get(cluster)) {
                writer.write(article+" "+cluster);
                writer.newLine();
            }
        }

        System.out.println("file name : "+input_file_name+" criteria : "+criterion_function+" clusters : "+num_clusters+"\nCriterion function value : "+result.criterion_value+ " Entropy : "+ result.entropy +
                " Purity : "+result.purity + " time taken : "+(end_time - start_time)/1000+" secs");

        writer.close();
    }
}
