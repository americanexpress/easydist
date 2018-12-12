# Copyright (c) 2019 American Express Travel Related Services Company, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

""" EasyDist's deployment GUI which integrates with AWS to
launch a cluster of GPU machines that can be used for
distributed training."""
# pylint: disable-msg=E0611
import sys
import boto3
from PyQt5.QtGui import QColor, QTextCursor
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QHBoxLayout, QLineEdit
from PyQt5.QtWidgets import QGridLayout, QComboBox, QPushButton, QVBoxLayout, QTextEdit
from deploy import AWS



class EasydistUi(QWidget):
    """ Class that creates UI Components"""
    def __init__(self):
        super().__init__()
        self.cluster = None
        self.ps_type = None
        self.worker_type = None
        self.worker_number = None
        self.iam_role = None

        self.init_ui()
        self.resize(600, 550)

    def init_ui(self):
        """ This method initialises the different parts of the UI and
        populates the drop down options and AWS IAM Roles"""

        #MAIN TITLE
        label = QLabel("EasyDist Machine Deployment", self)
        label.setStyleSheet("QLabel {font: 18pt;font:bold;}")
        hbox1 = QHBoxLayout()
        hbox1.addStretch(1)
        hbox1.addWidget(label)
        hbox1.addStretch(1)
        #-------------------

        #IAM ROLE DROP DOWN
        hbox2 = self.init_iam_ui()

        #Experiment Name
        name_label = QLabel("Experiment Name", self)
        name_label.move(70, 110)
        self.exp_name = QLineEdit(self)
        hbox3 = QHBoxLayout()
        hbox3.addWidget(name_label)
        hbox3.addWidget(self.exp_name)
        #-------------------

        #CLOUD PARAMTER LAYOUT DESIGN
        grid2 = QGridLayout()
        grid2.addLayout(hbox2, 0, 0)
        grid2.addLayout(hbox3, 1, 0)
        hbox6 = QHBoxLayout()
        hbox6.addStretch(1)
        hbox6.addLayout(grid2)
        hbox6.addStretch(2)
        #-----------------------------------

        hbox4 = self.init_machine_ui()
        #BUTTONS
        hbox5 = self.init_button_ui()

        #TEXTBOX FOR CONSOLE OUTPUT
        console_text_box = QTextEdit(self)
        console_text_box.setReadOnly(True)
        #Redirect STDOUT to the textbox
        sys.stdout = OutLog(console_text_box, sys.stdout)
        sys.stderr = OutLog(console_text_box, sys.stderr, QColor(255, 0, 0))
        #------------------------

        vbox = QVBoxLayout()
        vbox.addLayout(hbox1)
        vbox.addSpacing(20)
        vbox.addLayout(hbox6)
        vbox.addSpacing(30)
        vbox.addLayout(hbox4)
        vbox.addLayout(hbox5)
        vbox.addWidget(console_text_box)
        self.setLayout(vbox)
        self.setWindowTitle('EasyDist Machine Deployment')
        self.show()

    def init_iam_ui(self):
        """Fetches existing iam roles from your AWS account"""
        iam_label = QLabel("IAM Role", self)
        #Get Available Client Roles
        client = boto3.client('iam')
        roles = client.list_roles()
        all_roles = [key['RoleName'] for key in roles['Roles']]
        self.iam_role = QComboBox(self)
        for role in all_roles:
            self.iam_role.addItem(role)
        hbox2 = QHBoxLayout()
        hbox2.addWidget(iam_label)
        hbox2.addWidget(self.iam_role)
        return hbox2

    def init_button_ui(self):
        """ Creates all the UI Buttons"""
        deploy_button = QPushButton('Deploy Machines', self)
        deploy_button.clicked.connect(self.deploy_easydist)
        deploy_button.resize(deploy_button.sizeHint())

        login_button = QPushButton('Login', self)
        login_button.clicked.connect(self.login)
        login_button.resize(login_button.sizeHint())

        setup_button = QPushButton('Setup VMs', self)
        setup_button.clicked.connect(self.setup_vms)
        setup_button.resize(setup_button.sizeHint())

        quit_button = QPushButton('Terminate Machines', self)
        quit_button.clicked.connect(self.terminate_machines)
        quit_button.resize(quit_button.sizeHint())

        hbox5 = QHBoxLayout()
        hbox5.addStretch(1)
        hbox5.addWidget(deploy_button)
        hbox5.addWidget(login_button)
        hbox5.addWidget(setup_button)
        hbox5.addWidget(quit_button)
        hbox5.addStretch(1)
        return hbox5

    def init_machine_ui(self):
        """Creates drop down menus for VM Types"""
        #MACHINE DETAILS
        grid1 = QGridLayout()
        type_label = QLabel("Type", self)
        number_label = QLabel("Number", self)
        disk_label = QLabel("Local Storage (GB)", self)
        grid1.addWidget(type_label, 1, 0)
        grid1.addWidget(number_label, 2, 0)
        grid1.addWidget(disk_label, 3, 0)

        ps_machines = ['--select--', 'c3.2xlarge', 'c5.xlarge', 't2.nano', 'p2.xlarge']
        self.ps_type = QComboBox(self)
        for item in ps_machines:
            self.ps_type.addItem(item)
        ps_number = QComboBox(self)
        ps_number.addItem("1")
        ps_size = QLineEdit(self)
        ps_size.setText('75')
        grid1.addWidget(QLabel("Parameter Server", self), 0, 1)
        grid1.addWidget(self.ps_type, 1, 1)
        grid1.addWidget(ps_number, 2, 1)
        grid1.addWidget(ps_size, 3, 1)

        worker_machines = ['--select--', 'c3.2xlarge', 'p2.xlarge', 'p3.2xlarge', 'c5.xlarge', 't2.nano']
        self.worker_type = QComboBox(self)
        for item in worker_machines:
            self.worker_type.addItem(item)
        self.worker_number = QComboBox(self)
        for i in range(1, 17):
            self.worker_number.addItem(str(i))
        worker_size = QLineEdit(self)
        worker_size.setText('100')
        grid1.addWidget(QLabel("Worker", self), 0, 2)
        grid1.addWidget(self.worker_type, 1, 2)
        grid1.addWidget(self.worker_number, 2, 2)
        grid1.addWidget(worker_size, 3, 2)
        hbox4 = QHBoxLayout()
        hbox4.addStretch(1)
        hbox4.addLayout(grid1)
        hbox4.addStretch(1)
        return hbox4



    def login(self):
        """ Calls the deploy module's login method
        to log into parameter server of the cluster"""
        self.cluster.login()

    def setup_vms(self):
        """Calls the module's transfer method
        to setup easyDist and its dependencies
         on all the cluster's machines """
        self.cluster.transfer()

    def deploy_easydist(self):
        """ Uses GUI options and entries to launch
        the machines and organise the cluster on AWS"""
        worker_type = self.worker_type.currentText()
        worker_number = int(self.worker_number.currentText())

        ps_type = self.ps_type.currentText()

        exp_name = self.exp_name.text()
        iam_role = self.iam_role.currentText()
        print(exp_name, iam_role, worker_type, worker_number, ps_type)

        self.cluster = AWS(exp_name, iam_role, worker_type, worker_number, ps_type=ps_type)
        self.cluster.launch()

    def terminate_machines(self):
        """ Terminates all machines of the cluster"""
        self.cluster.terminate()


class OutLog:
    """Setups the TextEdit box to show the console output"""

    def __init__(self, edit, out=None, color=None):
        self.edit = edit
        self.out = out
        self.color = color

        

    def write(self, message):
        """ Writes the current console output to the Text Box"""
        """ Configures QT Text Box options """
        if self.color:
            text_color = self.edit.textColor()
            self.edit.setTextColor(self.color)

        self.edit.moveCursor(QTextCursor.End)
        self.edit.insertPlainText(message)

        if self.color:
            self.edit.setTextColor(text_color)

        if self.out:
            self.out.write(message)

        QApplication.processEvents()


if __name__ == '__main__':

    APP = QApplication(sys.argv)
    ex = EasydistUi()
    sys.exit(APP.exec_())
