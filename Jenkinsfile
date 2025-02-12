pipeline {
    agent any
    environment {
        TEMP_DIR="/tmp/${env.JOB_NAME}/${env.ghprbActualCommit}"
    } 
    stages {
        stage('Settin up env') {
            steps {
                echo "[[ Start virtual environment ]]"   
                sh """
                    echo "[ Current directory ] : " `pwd`
                    echo "[ Environment Variables ] "
                    env
# Each stage needs custom setting done again. By default /bin/python is used.
                    source /var/lib/jenkins/py3env/bin/activate
                    mkdir -p $TEMP_DIR
                    python -m venv $TEMP_DIR/venv
# activate new virtual env
                    source $TEMP_DIR/venv/bin/activate
                    echo "[ Python used ] : " `which python`
                    cd ${env.WORKSPACE}
                    echo "[ Install dependencies ]"
                    pip install -r requirements.txt
                    
                """
            }
        }
        stage('Run regression tests') {
            steps {
                echo '[[ Run pytest ]]' 
                sh """
# activate virtual environment again
                    source $TEMP_DIR/venv/bin/activate
                    echo "[ Python used ] : " `which python`
                    cd ${env.WORKSPACE}
                    echo "[ Installing ${env.JOB_NAME} ]"
                    python setup.py install --no-data
                    echo "[ Run test now ]"
                    cd qcore/test
                    pytest -s
                """
            }
        }
    }

    post {
        always {
            echo '[[ Tear down the environments ]]'
            sh """
                rm -rf $TEMP_DIR
            """
        }
    }
}
