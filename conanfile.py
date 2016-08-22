"""This makes aura installable through conan."""
from conans import ConanFile

class AuraConan(ConanFile):
    name = "aura"
    version = "0.1.0"
    license = "Boost"
    url = "https://github.com/sschaetz/aura"

    def source(self):
        self.run("git clone https://github.com/sschaetz/aura.git")

        # Here we should probably pin a tag/commit.
        #self.run("cd aura && git checkout 0.1.0")

    def package(self):
        self.copy("*.hpp", src="aura/include/", dst="include")
