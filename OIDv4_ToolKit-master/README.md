<h1> OID_v4 Toolkit</h1>

- Toolkit by [theAIGuysCode](https://github.com/theAIGuysCode/OIDv4_ToolKit) 

<br>

## Getting Started
    
Python is required.

1. Clone this respository

    ```sh
    git clone git clone https://github.com/EscVM/OIDv4_ToolKit.git
    ```

2. Install required packages with pip

   ```sh
   pip install -r requirements.txt
   ```
   
3. Download 1800 images of Person with the following command.

    ```sh
    python main.py downloader --classes Person --type_csv validation --limit 1800
    ```
    
    <strong>Arguments</strong>
    
    - **--classes**: Indicates the classes needed to be download
    - **--type_cvs**: Indicates the dataset type
    - **--limit**: Limit the number of dataset to be downloaded
    
    
    The algorithm will take care to download all the necessary files and build the directory structure like this:

    ```
    main_folder
    │   main.py
    │
    └───OID
        │   file011.txt
        │   file012.txt
        │
        └───csv_folder
        |    │   class-descriptions-boxable.csv
        |    │   validation-annotations-bbox.csv
        |
        └───Dataset
            |
            └─── train
                 |
                 └───Person
                      |
                      |000b65a36ad46f9e.jpg
                      |000e1dd786c8e433.jpg
                      |...
                      └───Label
                             |
                             |000b65a36ad46f9e.txt
                             |000e1dd786c8e433.txt
                             |...

    ```
    
### For YOLOv4

1. Run <strong> convert_annotations.py</strong>. This will generate  .txt annotation files in proper format for custom object detection with YOLOv4. The text files are generated in folder with images.

    ```sh
    python convert.annotations.py
    ```

### For MobilenetSSD

1. Download the file and place in the OIDv4_toolKit parent directory

    Link: [oid_to_pascal_voc_xml.py](https://gist.github.com/nilsfed/1dbf1cf397db50c90705daa6a81a8dec) 
    
2. Run the following command

    ```sh
    python oid_to_pascal_voc_xml.py
    ```
    
    The script will create directory called To_PASCAL_XML in the Dataset Subdirectories. These directories contain the XML files.


