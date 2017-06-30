# -*- mode: ruby -*-
# vi: set ft=ruby :
####

VAGRANTFILE_API_VERSION = "2"

Vagrant.configure(VAGRANTFILE_API_VERSION) do |config|

  config.vm.box = "bento/ubuntu-16.04"
  config.vm.box_url = "https://atlas.hashicorp.com/bento/boxes/ubuntu-16.04/versions/2.3.6/providers/virtualbox.box"

  config.vm.network "forwarded_port", guest: 8888, host: 8888

  config.vm.provider "virtualbox" do |v|
    v.name = "ubuntu"
    v.memory = 4096
    v.cpus = 6
  end
  
  config.vm.provision "shell", path: "provision.sh"

end
