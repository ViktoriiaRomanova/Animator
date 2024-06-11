from setuptools import setup

if __name__ == "__main__":

    with open("README.md") as f:
        readme = f.read()

    setup(
        # Metadata
        name = 'animator',
        version = '2.0.0',
        author = "Viktoriia Romanova",
        author_email = "khoviktoriya@yandex.ru",
        url= "https://github.com/ViktoriiaRomanova/Animator",
        description = "Training pipeline of Deep Learning models \
                       for the transformation of a person into a cartoon character",
        long_description=readme,
        packages = ['animator'],
        # Package info
        python_requires=">=3.10"
    )