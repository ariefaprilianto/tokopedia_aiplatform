from setuptools import setup, find_packages

setup(
    name='tokopedia_aiplatform',
    version='0.1.1',
    author='PT Tokopedia',
    author_email='engineering@tokopedia.com',
    description='Tokopedia AI Platform Package',
    packages=find_packages(include=['tokopedia_aiplatform', 'tokopedia_aiplatform.*']),
    install_requires=[
        'grpcio',
        'google-cloud-aiplatform',
        'langchain',      # Add langchain as a dependency
        'vertexai',       # Add vertexai as a dependency
        # Add other dependencies if needed
    ],
)
