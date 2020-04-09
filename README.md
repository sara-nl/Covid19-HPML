# Covid19-HPML
​
​
**Note: The COVID-Net models provided here are intended to be used as reference models that can be built upon and enhanced as new data becomes available. They are currently at a research stage and not yet intended as production-ready models (not meant for direct clinicial diagnosis), and we are working continuously to improve them as new data becomes available. Please do not use COVID-Net for self-diagnosis and seek help from your local health authorities.**
​
​
**SURFSara COVID-Net team: Valeriu Codreanu, Damian Podareanu, Ruben Hekster, Maxwell Cai, Joris Mollinga**



The world-wide pandemic response with regard to data and computer science includes so far analysing the spread of the virus, designing novel treatments or vaccines, understanding and predicting individual patient evolution as well as these implications on the healthcare system. 

As a cooperative association of the Dutch educational and research institutions, including the academic medical centers, SURF aims to support the efforts of all our members fighting against the COVID-19 pandemic. Besides offering a fast track for accessing the SURF infrastructure, we aim to offer a vision for the use of the national large-scale data, networking, compute services and expertise, in order to support researchers and to collaboratively work on this global problem. 

It is challenging to collect data centrally for analysis. Considering privacy constraints, we want to facilitate fast and secure network connections between data creators and national services and facilities. Moreover, we also need to make sure the right infrastructure is in place between the clinical and academic hospital members on one side and data and compute researchers on the other side. A practical flow could be to start from data placed securely at academic hospitals, transfer with a  XNAT-based data sharing architecture to central facilities and then enable researchers to utilise that data in computational flows. Data access can be regulated in a fine-grained manner and strong research groups coming from our various members can start analyzing clinical data. By using an open, federated way of data management and processing we can make sure that progress is achieved much faster than when aiming for isolated, close approaches, as the ones presented in the sections below. We are addressing here the issue of  scarcity of data, which is limiting the effectiveness of data-driven approaches. This is particularly related to diagnosis and prognosis tasks, but also true for drug design, where data is of utmost importance, especially since many analytics algorithms require annotated data for  supervised learning methods. During a crisis, it is obviously difficult to involve radiologists and other medical professionals in annotating data, as they are very busy with performing their other clinical duties. We want to assist them by providing easy-to-use tools, potentially hosted centrally.

Internally, SURFsara has the expertise to aid the research, development, and practical deployment of analytics algorithms based on prior or current collaborations. These cover social media, medical image, structural biology, and sound analytics.

For groups involved in drug discovery (such as Remco Havenith’s group in Groningen) we aim to use our high-performance computing expertise to assist quantum chemists and other relevant scientists to make the most of our large-scale infrastructure. 

In order to fight this pandemic, we propose to work together as a national research community, and the national data, networking, and compute infrastructure seems to be the most obvious platform for facilitating this. As a central infra facilitator, SURF can host data collection and annotation tools, can be part of the data sharing chain, connect relevant communities and help further research, development and adoption of COVID-19 analytics tools.

In the paragraphs below, we present several recent initiatives in imaging for diagnosis or prognosis, drug design, as well as in measuring and predicting the spread of the pandemic. 

Tags: data collection, data sharing, HPC resources, community support, drug design, streaming social media, medical image analysis, sound analytics

If there are any technical questions, please contact:
* valeriu.codreanu@surfsara.nl
* rubenh@surfsara.nl
* damian@surfsara.nl​
* joris.mollinga@surfara.nl
​
​
## Requirements
​
The main requirements are listed below:
​
* Tested with Tensorflow 1.13 and 1.15
* OpenCV 4.2.0
* Python 3.6
* Numpy
* OpenCV
* Scikit-Learn
* Matplotlib
​
Additional requirements to generate dataset:
​
* PyDicom
* Pandas
* Jupyter
