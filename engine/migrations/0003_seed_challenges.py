from django.db import migrations

CHALLENGES = [
    {
        "order": 1,
        "slug": "hello-chai",
        "title": "Hello, Chai!",
        "difficulty": "easy",
        "description": (
            "Print exactly the following text:\n\n"
            "Hello, ChaiScript!\n\n"
            "Use serveChai() to print output."
        ),
        "starter": "# Print Hello, ChaiScript!\n",
        "test_code": (
            "\nhot output_lines[-1].strip() == \"Hello, ChaiScript!\" then_chai\n"
            "    serveChai(\"ALL_TESTS_PASSED\")\nsip\n"
            "nahi_toh\n"
            "    serveChai(\"FAIL: expected Hello, ChaiScript!\")\nsip\n"
        ),
    },
    {
        "order": 2,
        "slug": "sum-two-cups",
        "title": "Sum Two Cups",
        "difficulty": "easy",
        "description": (
            "Declare two variables:\n"
            "  cup a = 10\n"
            "  cup b = 20\n\n"
            "Print their sum using serveChai().\n"
            "Expected output: 30"
        ),
        "starter": "# Declare a and b, then print their sum\n",
        "test_code": (
            "\nhot output_lines[-1].strip() == \"30\" then_chai\n"
            "    serveChai(\"ALL_TESTS_PASSED\")\nsip\n"
            "nahi_toh\n"
            "    serveChai(\"FAIL: expected 30\")\nsip\n"
        ),
    },
    {
        "order": 3,
        "slug": "brew-a-function",
        "title": "Brew a Function",
        "difficulty": "easy",
        "description": (
            "Write a function called double that takes one argument\n"
            "and returns it multiplied by 2.\n\n"
            "Then call it with 7 and print the result.\n"
            "Expected output: 14"
        ),
        "starter": (
            "# Write a brew called double\n"
            "brew double(n)\n"
            "    # your code here\n"
            "sip\n\n"
            "serveChai(double(7))\n"
        ),
        "test_code": (
            "\nhot output_lines[-1].strip() == \"14\" then_chai\n"
            "    serveChai(\"ALL_TESTS_PASSED\")\nsip\n"
            "nahi_toh\n"
            "    serveChai(\"FAIL: expected 14\")\nsip\n"
        ),
    },
    {
        "order": 4,
        "slug": "hot-or-cold",
        "title": "Hot or Cold?",
        "difficulty": "medium",
        "description": (
            "Given a variable:\n"
            "  cup temp = 85\n\n"
            "Print 'Hot' if temp > 60, otherwise print 'Cold'.\n"
            "Expected output: Hot"
        ),
        "starter": (
            "cup temp = 85\n"
            "# Check if hot or cold\n"
        ),
        "test_code": (
            "\nhot output_lines[-1].strip() == \"Hot\" then_chai\n"
            "    serveChai(\"ALL_TESTS_PASSED\")\nsip\n"
            "nahi_toh\n"
            "    serveChai(\"FAIL: expected Hot\")\nsip\n"
        ),
    },
    {
        "order": 5,
        "slug": "count-the-sips",
        "title": "Count the Sips",
        "difficulty": "medium",
        "description": (
            "Use a loop to print numbers 1 through 5, each on its own line.\n\n"
            "Expected output:\n"
            "1\n2\n3\n4\n5"
        ),
        "starter": "# Loop and print 1 to 5\n",
        "test_code": (
            "\ncup expected = [\"1\",\"2\",\"3\",\"4\",\"5\"]\n"
            "cup last5 = output_lines[-5:]\n"
            "hot [x.strip() for x in last5] == expected then_chai\n"
            "    serveChai(\"ALL_TESTS_PASSED\")\nsip\n"
            "nahi_toh\n"
            "    serveChai(\"FAIL: expected 1 2 3 4 5 on separate lines\")\nsip\n"
        ),
    },
]


def seed(apps, schema_editor):
    Challenge = apps.get_model('engine', 'Challenge')
    for c in CHALLENGES:
        Challenge.objects.get_or_create(slug=c['slug'], defaults=c)


def unseed(apps, schema_editor):
    Challenge = apps.get_model('engine', 'Challenge')
    Challenge.objects.filter(slug__in=[c['slug'] for c in CHALLENGES]).delete()


class Migration(migrations.Migration):
    dependencies = [
        ('engine', '0002_challenge_alter_sharedcode_id_challengesubmission_and_more'),
    ]

    operations = [
        migrations.RunPython(seed, unseed),
    ]