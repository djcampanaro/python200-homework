
# Part 1: Warmup - Cloud Concepts

## Cloud Concepts Q1

The core economic idea of cloud computing is that you rent the server space and pay f
or what you use rather than buying the hardware yourself to scale up. This is much more 
cost effective than purchasing ever more expensive hardware.

## Cloud Concepts Q2

The difference between vertical and horizontal scaling is vertical scaling involves 
purchasing more powerful hardware to handle the needed scaling while horizontal scaling 
spreads instances across devices as an application's traffic grows and reduces devices 
when apps have less traffic. You might choose to use vertical scaling when the data and 
application are needed and used locally and won't need to scale based on the number of 
users, such as a scientist working on a research project. Horizontal scaling would be used 
when the number of users could spike or fall depending on the amount of traffic/usage of 
a site or app. This would apply to a retail website that launches new lines and gets press.

## Cloud Concepts Q3

Gmail                             SaaS    Gmail is a service built on top of cloud elements 
and performs all tasks in the background, leaving the user with just an interface with 
which to interact.
Azure Virtual Machines            IaaS    Virtual Machines allows the user to set up their 
own environment and build on top.
AWS S3 (Simple Storage Service)   IaaS    S3 is a storage system that still requires the 
user to set up and manage.
GitHub Codespaces                 IaaS    Codespaces is also an environment on which the 
user can build their programs/applications
Snowflake                         PaaS    Snowflake is a service that manages the 
infrastructure, allowing the user to apply their own code.
Supabase                          PaaS    Supabase is also a service that uses the cloud 
in the background, allowing the user to interact with its interface.

IaaS is infrastructure as a service. An example of this is AWS EC2. The developer is 
responsible for setting up the os, building the environment, and handling security updates.

PaaS is platform as a service. An example is Azure App Service. The developer is responsible 
for writing the code and deploying the application.

SaaS is software as a service. An example is Dropbox. These services are for logging on 
and using. All the code and infrastructure is already handled and there is no development 
necessary.

## Cloud Concepts Q4

Managed platforms are optimized for databases and analytical workloads. They connect all 
the wires and work on top of the cloud providers' platforms rather than requiring a great 
amount of expertise and going through the lengthy and costly process of configuring all 
the elements of the cloud providers yourself. You gain more time and give more 
responsibilities to the managed platform, but you give up customization and control of how 
the cloud platforms are configured.

## Cloud Concepts Q5

A situation where the cloud may not be necessary is when setting up a prototype. When the 
dataset is manageable on a single system and a lot of compute power is not needed, the 
cloud is probably not the right tool.

# Part 2: Warmup - Cloud Landscape

## Cloud Landscape Q1

The three hyperscalers:
1. Amazon Web Services (AWS) - AWS offers the broadest service catalog of the three as it 
has been around the longest. It is good for large enterprises down to startups.
2. Google Cloud Platform (GCP) - GCP's strength is data and machine learning and would be 
a good fit for a company working on machine learning tools.
3. Microsoft Azure - Azure offers deep integration with Microsoft platforms and services 
and is greatly used in the non-profit and public sectors.

## Cloud Landscape Q2

Reasons CTD switched from Azure to Supabase:
Supabase is more accessible as each student can create and sign in to their own account 
rather than needing to be invited.
Supabase uses a relational database which is more transferable and fits in with the 
structure of the project we will build.
Supabase has stron pipeline coherence, making it easier to debug and inspect the pipeline 
stages.

When starting a new project, it would be a good idea to work out the scale and needs of 
the project before choosing a cloud model. Based on the size of the data, customization 
level, and scalability need, it is best to choose the cloud tool(s) that fit the specifics 
of the project to keep things simple and efficient.

## Cloud Landscape Q3

1. Object storage - AWS S3
2. ML Platform - Microsoft Azure ML
3. Serverless Compute - AWS Lambda
4. LLM API - GCP Vertex AI

## Cloud Landscape Q4


