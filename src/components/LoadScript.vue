<template>
  <div></div>
</template>
<script>
export default {
  props: {
    src: String,
  },
  emits: ["load"],
  mounted() {
    const script = document.createElement("script");
    script.setAttribute("type", "text/javascript");
    script.setAttribute("src", this.src);
    script.async = true;
    script.defer = true;
    script.onload = () => this.$emit("load", window.cv);

    document.body.appendChild(script);
  },
};
</script>